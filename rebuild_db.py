# rebuild_db.py
# Rebuilds logo_embeddings_clip.pkl using the
# lightweight extractor (same as Vercel uses)
# Run: python rebuild_db.py

import os
import pickle
import numpy as np
from PIL import Image

DATABASE_DIR    = "./authentic_logos"
EMBEDDINGS_FILE = "./logo_embeddings_clip.pkl"
SUPPORTED       = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class LightExtractor:
    def extract_from_pil(self, img):
        img = img.convert("RGB").resize((64, 64))
        arr = np.array(img).astype(np.float32)
        hist = []
        for ch in range(3):
            h, _ = np.histogram(arr[:,:,ch], bins=16, range=(0,255))
            hist.extend(h.tolist())
        gx = np.abs(np.diff(np.mean(arr,axis=2),axis=1)).mean()
        gy = np.abs(np.diff(np.mean(arr,axis=2),axis=0)).mean()
        vec = np.array(hist + [gx, gy, arr.mean()/255.0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def build():
    extractor = LightExtractor()
    db = {}

    if not os.path.isdir(DATABASE_DIR):
        print(f"Folder not found: {DATABASE_DIR}")
        return

    brands = [d for d in os.listdir(DATABASE_DIR)
              if os.path.isdir(os.path.join(DATABASE_DIR, d))]

    if not brands:
        print("No brand folders found.")
        return

    print(f"Building database for {len(brands)} brand(s)...")

    for brand in brands:
        folder = os.path.join(DATABASE_DIR, brand)
        images = [os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in SUPPORTED]
        if not images:
            print(f"  SKIP {brand} — no images")
            continue

        embeddings = []
        for path in images:
            try:
                img = Image.open(path).convert("RGB")
                emb = extractor.extract_from_pil(img)
                embeddings.append(emb)
            except Exception as e:
                print(f"  WARN skipped {path}: {e}")

        if embeddings:
            mean = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean)
            db[brand.lower()] = mean / norm if norm > 0 else mean
            print(f"  OK   {brand} ({len(embeddings)} image(s))")

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(db, f)

    print(f"\nDone! {len(db)} brand(s) saved to {EMBEDDINGS_FILE}")
    print("Now upload logo_embeddings_clip.pkl to GitHub and redeploy Vercel.")


if __name__ == "__main__":
    build()
