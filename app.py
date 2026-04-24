# ╔══════════════════════════════════════════════════════════════════╗
#  app.py — Fake Logo Detector (Vercel Version)
#  NO PyTorch, NO CLIP, NO torch imports
#  Uses SmartExtractor — fits within Vercel 500MB limit
# ╚══════════════════════════════════════════════════════════════════╝

import os, io, pickle, logging, time
from datetime import datetime
from functools import wraps
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

API_KEY          = os.environ.get("API_KEY", "change-me-secret")
ALLOWED_ORIGINS  = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
EMBEDDINGS_FILE  = os.environ.get("EMBEDDINGS_FILE", "./logo_embeddings_clip.pkl")
MAX_UPLOAD_MB    = int(os.environ.get("MAX_UPLOAD_MB", "10"))
RATE_LIMIT       = os.environ.get("RATE_LIMIT", "10 per minute")

SIMILARITY_THRESHOLD = 0.82
HIGH_CONF_THRESHOLD  = 0.90
LOW_CONF_THRESHOLD   = 0.65

IMAGE_SIGNATURES = {
    b'\xff\xd8\xff': 'jpeg',
    b'\x89PNG':      'png',
    b'GIF8':         'gif',
    b'RIFF':         'webp',
    b'BM':           'bmp',
}

app = Flask(__name__, static_folder='.')
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, origins=ALLOWED_ORIGINS)
limiter = Limiter(get_remote_address, app=app,
                  default_limits=[RATE_LIMIT], storage_uri="memory://")


# ══════════════════════════════════════════════
#  SMART EXTRACTOR — 256 dimensions
#  No PyTorch needed — pure Pillow + numpy
# ══════════════════════════════════════════════

class SmartExtractor:

    def extract_from_pil(self, img):
        img = img.convert("RGB").resize((128, 128), Image.LANCZOS)
        arr = np.array(img).astype(np.float32)
        features = np.concatenate([
            self._rgb_histogram(arr),
            self._hsv_histogram(img),
            self._edge_features(img),
            self._texture_features(arr),
            self._shape_features(arr),
            self._global_stats(arr),
        ])
        return self._l2_normalize(features)

    def _rgb_histogram(self, arr):
        hist = []
        for ch in range(3):
            h, _ = np.histogram(arr[:,:,ch], bins=16, range=(0,255))
            hist.extend((h / (h.sum() + 1e-8)).tolist())
        return np.array(hist, dtype=np.float32)

    def _hsv_histogram(self, img):
        try:
            hsv_arr = np.array(img.convert("HSV")).astype(np.float32)
        except Exception:
            hsv_arr = np.array(img).astype(np.float32)
        hist = []
        for ch in range(3):
            h, _ = np.histogram(hsv_arr[:,:,ch], bins=16, range=(0,255))
            hist.extend((h / (h.sum() + 1e-8)).tolist())
        return np.array(hist, dtype=np.float32)

    def _edge_features(self, img):
        gray  = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges).astype(np.float32)
        h, w  = edge_arr.shape
        rh, rw = h // 4, w // 4
        features = []
        for i in range(4):
            for j in range(4):
                region = edge_arr[i*rh:(i+1)*rh, j*rw:(j+1)*rw]
                features.append(region.mean() / 255.0)
                features.append(region.std() / 255.0)
        return np.array(features, dtype=np.float32)

    def _texture_features(self, arr):
        gray = np.mean(arr, axis=2)
        features = []
        h, w = gray.shape
        gh, gw = h // 8, w // 8
        for i in range(8):
            for j in range(8):
                region = gray[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                mean   = region.mean()
                std    = region.std()
                contrast = std / (mean + 1e-8)
                features.append(min(contrast, 5.0) / 5.0)
        return np.array(features, dtype=np.float32)

    def _shape_features(self, arr):
        gray = np.mean(arr, axis=2)
        h, w = gray.shape
        features = []
        for i in range(8):
            band = gray[i*h//8:(i+1)*h//8, :]
            features.append(band.mean() / 255.0)
        for j in range(8):
            band = gray[:, j*w//8:(j+1)*w//8]
            features.append(band.mean() / 255.0)
        diag1 = np.diag(gray).mean() / 255.0
        diag2 = np.diag(np.fliplr(gray)).mean() / 255.0
        features.extend([diag1, diag2])
        center = gray[h//4:3*h//4, w//4:3*w//4].mean()
        border = gray.mean()
        features.append(center / (border + 1e-8))
        left  = gray[:, :w//2]
        right = np.fliplr(gray[:, w//2:])
        sym_h = 1.0 - np.abs(left - right).mean() / 255.0
        features.append(sym_h)
        top    = gray[:h//2, :]
        bottom = np.flipud(gray[h//2:, :])
        sym_v  = 1.0 - np.abs(top - bottom).mean() / 255.0
        features.append(sym_v)
        while len(features) < 32:
            features.append(0.0)
        return np.array(features[:32], dtype=np.float32)

    def _global_stats(self, arr):
        features = []
        for ch in range(3):
            ch_data = arr[:,:,ch] / 255.0
            features.extend([
                ch_data.mean(),
                ch_data.std(),
                float(np.percentile(ch_data, 25)),
                float(np.percentile(ch_data, 75)),
                float(np.percentile(ch_data, 10)),
                float(np.percentile(ch_data, 90)),
            ])
        gray = np.mean(arr, axis=2) / 255.0
        features.append(gray.mean())
        features.append(gray.std())
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        rg = np.abs(r.astype(float) - g.astype(float))
        yb = np.abs(0.5*(r.astype(float)+g.astype(float)) - b.astype(float))
        colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + \
                       0.3*np.sqrt(rg.mean()**2 + yb.mean()**2)
        features.append(min(colorfulness, 200.0) / 200.0)
        h, w = arr.shape[:2]
        features.append(w / (h + 1e-8))
        while len(features) < 32:
            features.append(0.0)
        return np.array(features[:32], dtype=np.float32)

    @staticmethod
    def _l2_normalize(vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


extractor = SmartExtractor()


# ══════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════

def load_database():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            db = pickle.load(f)
        logger.info(f"DB loaded — {len(db)} brands")
        return db
    logger.warning(f"No DB at {EMBEDDINGS_FILE}")
    return {}

database = load_database()

def save_database():
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(database, f)


# ══════════════════════════════════════════════
#  FILE VALIDATION
# ══════════════════════════════════════════════

def detect_image_type(raw):
    for sig, name in IMAGE_SIGNATURES.items():
        if raw[:len(sig)] == sig:
            return name
    return None

def validate_image(raw):
    if not raw:
        raise ValueError("Empty file.")
    if not detect_image_type(raw):
        raise ValueError("Invalid file type. Only jpg, png, webp, bmp allowed.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise ValueError("Corrupted or invalid image.")
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata()))
    return clean.convert("RGB")


# ══════════════════════════════════════════════
#  API KEY
# ══════════════════════════════════════════════

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if not key:
            return jsonify({"success": False, "error": "API key required."}), 401
        if key != API_KEY:
            return jsonify({"success": False, "error": "Invalid API key."}), 401
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════

@app.before_request
def before():
    g.start = time.time()

@app.after_request
def after(response):
    ms = round((time.time() - g.start) * 1000, 1)
    logger.info(f"{request.method} {request.path} → {response.status_code} ({ms}ms)")
    return response


# ══════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"success": False, "error": "Bad request."}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"success": False, "error": "Unauthorized."}), 401

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": f"Not found: {request.path}"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": f"File too large. Max {MAX_UPLOAD_MB}MB."}), 413

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"success": False, "error": "Too many requests. Try again later."}), 429

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500: {e}")
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ══════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def run_detection(img):
    if not database:
        return {
            "verdict": "UNKNOWN", "confidence": "LOW",
            "similarity_score": 0.0, "matched_brand": None,
            "message": "Database empty. Run rebuild_db.py first."
        }
    query = extractor.extract_from_pil(img)
    best_brand, best_score = None, -1.0
    for brand, stored in database.items():
        if stored.shape != query.shape:
            continue
        s = cosine_sim(query, stored)
        if s > best_score:
            best_score, best_brand = s, brand
    if best_brand is None:
        return {
            "verdict": "UNKNOWN", "confidence": "LOW",
            "similarity_score": 0.0, "matched_brand": None,
            "message": "Database needs rebuild. Run rebuild_db.py"
        }
    best_score = round(best_score, 4)
    if best_score >= HIGH_CONF_THRESHOLD:
        v, c, m = "AUTHENTIC", "HIGH",   f"Very close match to '{best_brand}'."
    elif best_score >= SIMILARITY_THRESHOLD:
        v, c, m = "AUTHENTIC", "MEDIUM", f"Matches '{best_brand}' with moderate confidence."
    elif best_score >= LOW_CONF_THRESHOLD:
        v, c, m = "FAKE",      "MEDIUM", f"Resembles '{best_brand}' but similarity too low."
    else:
        v, c, m = "FAKE",      "HIGH",   "No match found in database."
    return {
        "verdict": v, "confidence": c,
        "similarity_score": best_score,
        "matched_brand": best_brand,
        "message": m,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════

@app.route("/")
@limiter.exempt
def index():
    return send_from_directory('.', 'index.html')

@app.route("/health")
@limiter.exempt
def health():
    return jsonify({
        "status": "ok",
        "extractor": "SmartExtractor-256dim",
        "brands_loaded": len(database),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route("/detect", methods=["POST"])
@limiter.limit("10 per minute")
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"success": False,
                "error": "No image. Send multipart/form-data with field 'image'."}), 400
        raw = request.files["image"].read()
        img = validate_image(raw)
        result = run_detection(img)
        logger.info(f"DETECT {result['verdict']} score={result['similarity_score']}")
        return jsonify({"success": True, "result": result})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"DETECT error: {e}")
        return jsonify({"success": False, "error": "Detection failed."}), 500

@app.route("/brands")
@limiter.limit("30 per minute")
def brands():
    return jsonify({
        "success": True,
        "count": len(database),
        "brands": sorted(database.keys())
    })

@app.route("/add-brand", methods=["POST"])
@require_api_key
@limiter.limit("20 per hour")
def add_brand():
    try:
        name = request.form.get("brand_name", "").strip().lower()
        if not name:
            return jsonify({"success": False, "error": "brand_name required."}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required."}), 400
        raw = request.files["image"].read()
        img = validate_image(raw)
        database[name] = extractor.extract_from_pil(img)
        save_database()
        return jsonify({"success": True, "message": f"'{name}' added.",
                        "total_brands": len(database)}), 201
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": "Failed."}), 500

@app.route("/brand/<brand_name>", methods=["DELETE"])
@require_api_key
def remove_brand(brand_name):
    key = brand_name.lower().strip()
    if key not in database:
        return jsonify({"success": False, "error": f"'{key}' not found."}), 404
    del database[key]
    save_database()
    return jsonify({"success": True, "message": f"'{key}' removed.",
                    "remaining": len(database)})

if __name__ == "__main__":
    logger.info("Starting local dev server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
