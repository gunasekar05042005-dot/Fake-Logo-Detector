import os
import pickle
import numpy as np
from PIL import Image
import gradio as gr
import torch
import open_clip

# ── Config ────────────────────────────────────
EMBEDDINGS_FILE      = "logo_embeddings_clip.pkl"
SIMILARITY_THRESHOLD = 0.80
HIGH_CONF_THRESHOLD  = 0.88
LOW_CONF_THRESHOLD   = 0.60

# ── Load CLIP ─────────────────────────────────
print("Loading CLIP ViT-B/32...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.eval().to(device)
print(f"CLIP ready on {device}")

# ── Load database ─────────────────────────────
def load_db():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            db = pickle.load(f)
        print(f"Loaded {len(db)} brands")
        return db
    print("No database file found")
    return {}

database = load_db()

# ── Extract CLIP embedding ─────────────────────
def extract(img):
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    vec = features.squeeze().float().cpu().numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# ── Cosine similarity ──────────────────────────
def cosine_sim(a, b):
    return float(np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ── Detection ─────────────────────────────────
def detect(image):
    if image is None:
        return "⬆ Upload a logo image first", "", "", ""

    if not database:
        return "❓ Database empty", "No brands loaded.", "0%", "None"

    # Rebuild database if dimension mismatch
    query = extract(image)
    best_brand, best_score = None, -1.0

    for brand, stored in database.items():
        # Check dimension match
        if stored.shape != query.shape:
            return (
                "⚠ Database mismatch",
                f"Database has {stored.shape[0]}-dim vectors but CLIP produces "
                f"{query.shape[0]}-dim vectors. Please rebuild database with "
                f"fake_logo_detector_allinone_1.py --build and re-upload pkl file.",
                "0%",
                "None"
            )
        s = cosine_sim(query, stored)
        if s > best_score:
            best_score, best_brand = s, brand

    best_score = round(best_score, 4)
    pct        = round(best_score * 100, 1)
    brand_cap  = best_brand.capitalize() if best_brand else "None"

    if best_score >= HIGH_CONF_THRESHOLD:
        verdict = "✅ AUTHENTIC — HIGH confidence"
        msg     = f"Very close CLIP match to '{brand_cap}'. Score: {pct}%\nThis logo is genuine."
    elif best_score >= SIMILARITY_THRESHOLD:
        verdict = "✅ AUTHENTIC — MEDIUM confidence"
        msg     = f"Matches '{brand_cap}' with moderate confidence. Score: {pct}%"
    elif best_score >= LOW_CONF_THRESHOLD:
        verdict = "❌ FAKE — MEDIUM confidence"
        msg     = f"Resembles '{brand_cap}' but similarity too low. Score: {pct}%\nLikely counterfeit."
    else:
        verdict = "❌ FAKE — HIGH confidence"
        msg     = f"No match in database. Score: {pct}%\nDefinitely fake or unknown brand."

    return verdict, msg, f"{pct}%", brand_cap

# ── Brand list ─────────────────────────────────
brand_list = " • ".join(
    b.capitalize() for b in sorted(database.keys())
) if database else "No brands loaded — upload logo_embeddings_clip.pkl"

# ── Gradio UI ──────────────────────────────────
with gr.Blocks(
    title="🐦 Fake Logo Detector",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Nunito"),
    ),
    css="""
    .gradio-container{max-width:720px!important;margin:0 auto!important}
    footer{display:none!important}
    #verdict textarea{
        font-size:22px!important;
        font-weight:800!important;
        text-align:center!important;
        color:#ff6b00!important;
    }
    """
) as demo:

    # Header
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 20px;
         background:linear-gradient(135deg,#1a0a00,#2d1500);
         border-radius:16px;margin-bottom:16px;
         border:1px solid rgba(255,107,0,0.3)">
      <div style="font-size:60px;margin-bottom:8px">🐦</div>
      <h1 style="font-size:28px;font-weight:900;
          color:#ff6b00;margin:0;letter-spacing:1px">
        FAKE LOGO DETECTOR
      </h1>
      <p style="color:#ffaa55;font-size:13px;margin-top:6px;font-weight:600">
        Angry Birds Edition — Powered by OpenAI CLIP ViT-B/32
      </p>
      <div style="display:inline-block;margin-top:10px;
           background:#ff6b00;color:#fff;
           font-size:11px;font-weight:700;padding:4px 14px;
           border-radius:99px;letter-spacing:1px">
        FULL AI ACCURACY
      </div>
    </div>
    """)

    with gr.Row(equal_height=True):

        # Left column — upload
        with gr.Column(scale=1):
            gr.HTML("""
            <p style="font-size:13px;font-weight:700;
               color:#ffaa55;margin-bottom:8px">
              📁 Upload Logo Image
            </p>
            """)
            image_input = gr.Image(
                type="pil",
                label="",
                height=240,
                show_label=False,
            )
            detect_btn = gr.Button(
                "🎯 FIRE! Detect Logo",
                variant="primary",
                size="lg",
            )
            clear_btn = gr.Button(
                "✕ Clear",
                variant="secondary",
                size="sm",
            )

        # Right column — results
        with gr.Column(scale=1):
            gr.HTML("""
            <p style="font-size:13px;font-weight:700;
               color:#ffaa55;margin-bottom:8px">
              📊 Detection Result
            </p>
            """)
            verdict_out = gr.Textbox(
                label="Verdict",
                interactive=False,
                lines=2,
                elem_id="verdict",
                placeholder="Result will appear here...",
            )
            message_out = gr.Textbox(
                label="Details",
                interactive=False,
                lines=3,
                placeholder="Details will appear here...",
            )
            with gr.Row():
                score_out = gr.Textbox(
                    label="📈 Similarity",
                    interactive=False,
                )
                brand_out = gr.Textbox(
                    label="🏷 Matched brand",
                    interactive=False,
                )

    # Detect button action
    detect_btn.click(
        fn=detect,
        inputs=[image_input],
        outputs=[verdict_out, message_out, score_out, brand_out],
    )

    # Clear button action
    clear_btn.click(
        fn=lambda: (None, "", "", "", ""),
        outputs=[image_input, verdict_out, message_out, score_out, brand_out],
    )

    # Brands section
    gr.HTML(f"""
    <div style="margin-top:16px;padding:16px;
         background:rgba(255,107,0,0.06);
         border-radius:14px;
         border:1px solid rgba(255,107,0,0.2)">
      <p style="font-size:12px;color:#ff9944;
         font-weight:700;margin-bottom:8px">
        🐷 {len(database)} BRANDS IN DATABASE
      </p>
      <p style="font-size:12px;color:#888;line-height:2">
        {brand_list}
      </p>
    </div>
    """)

    # Footer
    gr.HTML("""
    <div style="text-align:center;margin-top:16px;
         font-size:12px;color:#555;padding-bottom:8px">
      Built by
      <strong style="color:#ff6b00">Gunasekar</strong>
      &nbsp;|&nbsp; BCA Final Year, Tamil Nadu
      &nbsp;|&nbsp;
      <a href="https://github.com/gunasekar05042005-dot/Fake-Logo-Detector"
         style="color:#ff6b00;text-decoration:none"
         target="_blank">GitHub ↗</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
