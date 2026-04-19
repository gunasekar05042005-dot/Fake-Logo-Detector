# ╔══════════════════════════════════════════════════════════════════╗
#  FAKE LOGO DETECTION SYSTEM — CLIP Edition (All-in-One)
#
#  Install:
#    pip install open-clip-torch Pillow scikit-learn flask flask-cors tqdm
#
#  Commands:
#    python fake_logo_detector_allinone.py --build
#    python fake_logo_detector_allinone.py --detect logo.png
#    python fake_logo_detector_allinone.py --list
#    python fake_logo_detector_allinone.py --add nike nike.png
#    python fake_logo_detector_allinone.py --compare a.png b.png
#    python fake_logo_detector_allinone.py --api
# ╚══════════════════════════════════════════════════════════════════╝

import os, io, sys, pickle, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import torch
import open_clip

from flask import Flask, request, jsonify
from flask_cors import CORS


# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════

DATABASE_DIR         = "./authentic_logos"
EMBEDDINGS_FILE      = "./logo_embeddings_clip.pkl"

CLIP_MODEL_NAME      = "ViT-B-32"
CLIP_PRETRAINED      = "openai"
EMBEDDING_DIM        = 512

SIMILARITY_THRESHOLD = 0.80
HIGH_CONF_THRESHOLD  = 0.88
LOW_CONF_THRESHOLD   = 0.60

API_HOST             = "0.0.0.0"
API_PORT             = 5000
MAX_UPLOAD_MB        = 10


# ══════════════════════════════════════════════
#  FEATURE EXTRACTOR
# ══════════════════════════════════════════════

class FeatureExtractor:

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[CLIP] Loading {CLIP_MODEL_NAME} on {self.device} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        self.model.eval().to(self.device)
        print(f"[CLIP] Ready.")

    def extract(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert("RGB")
        return self._embed(img)

    def extract_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        return self._embed(pil_image.convert("RGB"))

    def extract_from_bytes(self, raw: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self._embed(img)

    def _embed(self, img: Image.Image) -> np.ndarray:
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        vec = features.squeeze().float().cpu().numpy()
        return self._l2_normalize(vec)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ══════════════════════════════════════════════
#  LOGO DATABASE
# ══════════════════════════════════════════════

class LogoDatabase:

    SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
        self.db: Dict[str, np.ndarray] = {}

    def build_from_folder(self, logos_dir: str = DATABASE_DIR) -> None:
        if not os.path.isdir(logos_dir):
            print(f"[Database] Folder not found: {logos_dir}")
            print(f"[Database] Create it first:")
            print(f"           {logos_dir}\\<brand_name>\\logo.png")
            return
        brands = [d for d in os.listdir(logos_dir)
                  if os.path.isdir(os.path.join(logos_dir, d))]
        if not brands:
            print(f"[Database] No brand sub-folders found in {logos_dir}")
            return
        print(f"[Database] Found {len(brands)} brand(s). Building ...")
        for brand in tqdm(brands, desc="Indexing"):
            self._index_brand(brand, os.path.join(logos_dir, brand))
        print(f"[Database] Done — {len(self.db)} brand(s) indexed.")

    def _index_brand(self, brand: str, folder: str) -> None:
        images = [os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in self.SUPPORTED]
        if not images:
            return
        embeddings = []
        for path in images:
            try:
                embeddings.append(self.extractor.extract(path))
            except Exception as e:
                print(f"  [warn] Skipped {path}: {e}")
        if embeddings:
            mean = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean)
            self.db[brand.lower()] = mean / norm if norm > 0 else mean

    def add_brand(self, brand: str, image_path: str) -> None:
        self.db[brand.lower()] = self.extractor.extract(image_path)
        print(f"[Database] Added '{brand}'.")

    def add_brand_from_pil(self, brand: str, pil_image: Image.Image) -> None:
        self.db[brand.lower()] = self.extractor.extract_from_pil(pil_image)
        print(f"[Database] Added '{brand}'.")

    def remove_brand(self, brand: str) -> bool:
        key = brand.lower()
        if key in self.db:
            del self.db[key]
            return True
        return False

    def save(self, path: str = EMBEDDINGS_FILE) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.db, f)
        print(f"[Database] Saved {len(self.db)} brand(s) → {path}")

    def load(self, path: str = EMBEDDINGS_FILE) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.db = pickle.load(f)
        print(f"[Database] Loaded {len(self.db)} brand(s) from {path}")
        return True

    def list_brands(self) -> List[str]:
        return sorted(self.db.keys())

    def __len__(self):
        return len(self.db)


# ══════════════════════════════════════════════
#  DETECTION RESULT
# ══════════════════════════════════════════════

@dataclass
class DetectionResult:
    verdict: str
    confidence: str
    similarity_score: float
    matched_brand: Optional[str]
    message: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        icon = "✓" if self.verdict == "AUTHENTIC" else "✗"
        line = "-" * 52
        return (
            f"\n{line}\n"
            f"  {icon}  Verdict     : {self.verdict} ({self.confidence} confidence)\n"
            f"     Score      : {self.similarity_score:.4f}  "
            f"(threshold: {SIMILARITY_THRESHOLD})\n"
            f"     Best match : {self.matched_brand or 'None'}\n"
            f"     {self.message}\n"
            f"{line}"
        )


# ══════════════════════════════════════════════
#  DETECTOR
# ══════════════════════════════════════════════

class LogoDetector:

    def __init__(self, extractor: FeatureExtractor, database: LogoDatabase):
        self.extractor = extractor
        self.db        = database

    def detect(self, image_path: str) -> DetectionResult:
        return self._compare(self.extractor.extract(image_path))

    def detect_from_pil(self, pil_image: Image.Image) -> DetectionResult:
        return self._compare(self.extractor.extract_from_pil(pil_image))

    def _compare(self, query: np.ndarray) -> DetectionResult:
        if len(self.db) == 0:
            return DetectionResult(
                verdict="UNKNOWN", confidence="LOW",
                similarity_score=0.0, matched_brand=None,
                message="Database is empty. Run --build first.",
            )
        brand, score = self._best_match(query)
        return self._verdict(brand, score)

    def _best_match(self, q: np.ndarray):
        q2d = q.reshape(1, -1)
        best_b, best_s = None, -1.0
        for brand, stored in self.db.db.items():
            s = float(cosine_similarity(q2d, stored.reshape(1, -1))[0][0])
            if s > best_s:
                best_s, best_b = s, brand
        return best_b, best_s

    def _verdict(self, brand: str, score: float) -> DetectionResult:
        score = round(score, 4)
        if score >= HIGH_CONF_THRESHOLD:
            return DetectionResult("AUTHENTIC", "HIGH",   score, brand,
                f"Very close CLIP match to '{brand}'.")
        elif score >= SIMILARITY_THRESHOLD:
            return DetectionResult("AUTHENTIC", "MEDIUM", score, brand,
                f"Matches '{brand}' with moderate confidence.")
        elif score >= LOW_CONF_THRESHOLD:
            return DetectionResult("FAKE",      "MEDIUM", score, brand,
                f"Resembles '{brand}' but similarity too low. Likely counterfeit.")
        else:
            return DetectionResult("FAKE",      "HIGH",   score, brand,
                "No match found in authentic database.")


# ══════════════════════════════════════════════
#  FLASK API
# ══════════════════════════════════════════════

def create_app(extractor, database, detector):
    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    def _get_image():
        if "image" not in request.files:
            raise ValueError("No image. Send as multipart/form-data field 'image'.")
        raw = request.files["image"].read()
        if not raw:
            raise ValueError("Empty file.")
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "brands": len(database),
                        "device": extractor.device})

    @app.route("/detect", methods=["POST"])
    def detect():
        try:
            result = detector.detect_from_pil(_get_image())
            return jsonify({"success": True, "result": result.to_dict()})
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/brands")
    def brands():
        return jsonify({"brands": database.list_brands(), "count": len(database)})

    @app.route("/add-brand", methods=["POST"])
    def add_brand():
        try:
            name = request.form.get("brand_name", "").strip().lower()
            if not name:
                return jsonify({"success": False, "error": "brand_name required"}), 400
            database.add_brand_from_pil(name, _get_image())
            database.save()
            return jsonify({"success": True, "message": f"'{name}' added."}), 201
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/brand/<n>", methods=["DELETE"])
    def remove_brand(name):
        if database.remove_brand(name):
            database.save()
            return jsonify({"success": True, "message": f"'{name}' removed."})
        return jsonify({"success": False, "error": f"'{name}' not found."}), 404

    return app


# ══════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════

def run_cli(extractor, database, detector):
    parser = argparse.ArgumentParser(
        description="Fake Logo Detector — CLIP edition",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fake_logo_detector_allinone.py --build\n"
            "  python fake_logo_detector_allinone.py --detect logo.png\n"
            "  python fake_logo_detector_allinone.py --add nike nike.png\n"
            "  python fake_logo_detector_allinone.py --list\n"
            "  python fake_logo_detector_allinone.py --compare a.png b.png\n"
            "  python fake_logo_detector_allinone.py --api\n"
        ),
    )
    parser.add_argument("--build",   action="store_true",
                        help="Build DB from authentic_logos folder")
    parser.add_argument("--detect",  metavar="IMAGE",
                        help="Detect a single logo image")
    parser.add_argument("--add",     nargs=2, metavar=("BRAND", "FILE"),
                        help="Add one brand manually")
    parser.add_argument("--list",    action="store_true",
                        help="List all brands in database")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two logo images directly")
    parser.add_argument("--api",     action="store_true",
                        help="Start Flask REST API server")
    args = parser.parse_args()

    if args.build:
        database.build_from_folder()
        if len(database) > 0:
            database.save()
        return

    if args.list:
        brands = database.list_brands()
        if brands:
            print(f"\n{len(brands)} brand(s) in database:")
            for b in brands:
                print(f"  - {b}")
        else:
            print("Database is empty. Run --build first.")
        return

    if args.add:
        brand, file = args.add
        database.add_brand(brand, file)
        database.save()
        return

    if args.detect:
        if not os.path.exists(args.detect):
            print(f"File not found: {args.detect}")
            sys.exit(1)
        if len(database) == 0:
            print("Database is empty. Run --build first.")
            sys.exit(1)
        print(detector.detect(args.detect))
        return

    if args.compare:
        a_path, b_path = args.compare
        for p in (a_path, b_path):
            if not os.path.exists(p):
                print(f"File not found: {p}")
                sys.exit(1)
        emb_a = extractor.extract(a_path)
        emb_b = extractor.extract(b_path)
        score = float(cosine_similarity(
            emb_a.reshape(1,-1), emb_b.reshape(1,-1))[0][0])
        label = (
            "Very similar  — likely same brand"    if score >= HIGH_CONF_THRESHOLD else
            "Similar       — same brand, variant"  if score >= SIMILARITY_THRESHOLD else
            "Weak match    — possible counterfeit" if score >= LOW_CONF_THRESHOLD  else
            "No match      — different or fake"
        )
        print(f"\n  Image A : {a_path}")
        print(f"  Image B : {b_path}")
        print(f"  Score   : {score:.4f}")
        print(f"  Result  : {label}\n")
        return

    if args.api:
        app = create_app(extractor, database, detector)
        print(f"\n[API] Running on http://localhost:{API_PORT}")
        print("[API] POST /detect | GET /brands | POST /add-brand | GET /health\n")
        app.run(host=API_HOST, port=API_PORT, debug=False)
        return

    parser.print_help()


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 52)
    print("  Fake Logo Detector  —  CLIP Edition")
    print(f"  Model : {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    print("=" * 52)

    extractor = FeatureExtractor()
    database  = LogoDatabase(extractor)

    if not database.load():
        print("[Main] No saved DB found.")

    detector = LogoDetector(extractor, database)
    run_cli(extractor, database, detector)
