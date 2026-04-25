import os
import pickle
import numpy as np
from PIL import Image, ImageFilter

DATABASE_DIR    = "./authentic_logos"
EMBEDDINGS_FILE = "./logo_embeddings_clip.pkl"
SUPPORTED       = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def build():
    extractor = SmartExtractor()
    db = {}

    if not os.path.isdir(DATABASE_DIR):
        print("Folder not found: " + DATABASE_DIR)
        return

    brands = [d for d in os.listdir(DATABASE_DIR)
              if os.path.isdir(os.path.join(DATABASE_DIR, d))]

    if not brands:
        print("No brand folders found.")
        return

    print("Building database for " + str(len(brands)) + " brand(s)...")

    for brand in brands:
        folder = os.path.join(DATABASE_DIR, brand)
        images = [os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in SUPPORTED]
        if not images:
            print("  SKIP " + brand + " — no images")
            continue
        embeddings = []
        for path in images:
            try:
                img = Image.open(path).convert("RGB")
                emb = extractor.extract_from_pil(img)
                embeddings.append(emb)
            except Exception as e:
                print("  WARN skipped " + path + ": " + str(e))
        if embeddings:
            mean = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean)
            db[brand.lower()] = mean / norm if norm > 0 else mean
            print("  OK   " + brand + " (" + str(len(embeddings)) + " image)")

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(db, f)

    print("")
    print("Done! " + str(len(db)) + " brand(s) saved to " + EMBEDDINGS_FILE)

if __name__ == "__main__":
    build()
