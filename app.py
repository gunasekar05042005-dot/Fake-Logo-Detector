import os
import pickle
import numpy as np
from PIL import Image
import gradio as gr
import torch
import open_clip

EMBEDDINGS_FILE      = "logo_embeddings_clip.pkl"
CLIP_MODEL_NAME      = "ViT-B-32"
CLIP_PRETRAINED      = "openai"
SIMILARITY_THRESHOLD = 0.80
HIGH_CONF_THRESHOLD  = 0.88
LOW_CONF_THRESHOLD   = 0.60

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
)
model.eval().to(device)
print(f"CLIP ready on {device}")

def load_db():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            db = pickle.load(f)
        print(f"Database loaded — {len(db)} brands")
        return db
    print("No database found")
    return {}

database = load_db()

def extract(img):
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    vec = features.squeeze().float().cpu().numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def run_detection(image):
    if image is None:
        return "Please upload a logo image.", "", "", ""
    if not database:
        return "❓ UNKNOWN", "Database empty — no brands loaded.", "0.0%", "None"

    query = extract(image)
    best_brand, best_score = None, -1.0
    for brand, stored in database.items():
        s = cosine_sim(query, stored)
        if s > best_score:
            best_score, best_brand = s, brand

    best_score = round(best_score, 4)
    pct        = round(best_score * 100, 1)
    brand_name = best_brand.capitalize() if best_brand else "None"

    if best_score >= HIGH_CONF_THRESHOLD:
        verdict = "✅ AUTHENTIC — HIGH confidence"
        msg     = f"Very close CLIP match to '{brand_name}'. Score: {pct}%"
    elif best_score >= SIMILARITY_THRESHOLD:
        verdict = "✅ AUTHENTIC — MEDIUM confidence"
        msg     = f"Matches '{brand_name}' with moderate confidence. Score: {pct}%"
    elif best_score >= LOW_CONF_THRESHOLD:
        verdict = "❌ FAKE — MEDIUM confidence"
        msg     = f"Resembles '{brand_name}' but similarity too low. Score: {pct}%"
    else:
        verdict = "❌ FAKE — HIGH confidence"
        msg     = f"No match found in database. Score: {pct}%"

    return verdict, msg, f"{pct}%", brand_name

brand_list = ", ".join(
    b.capitalize() for b in sorted(database.keys())
) if database else "No brands loaded"

with gr.Blocks(
    title="Fake Logo Detector",
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="red",
        font=gr.themes.GoogleFont("Nunito"),
    ),
    css="""
    .gradio-container{max-width:700px!important;margin:0 auto!important}
    footer{display:none!important}
    """
) as app:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 16px">
      <div style="font-size:56px;margin-bottom:8px">🐦</div>
      <h1 style="font-size:30px;font-weight:800;color:#ff6b00;margin:0">
        Fake Logo Detector
      </h1>
      <p style="color:#aaa;font-size:14px;margin-top:6px">
        Powered by OpenAI CLIP ViT-B/32 — Full AI accuracy
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Logo Image",
                height=260,
            )
            detect_btn = gr.Button(
                "🎯 FIRE! Detect Logo",
                variant="primary",
                size="lg",
            )

        with gr.Column():
            verdict_out = gr.Textbox(
                label="Verdict",
                interactive=False,
                lines=2,
            )
            message_out = gr.Textbox(
                label="Details",
                interactive=False,
                lines=3,
            )
            with gr.Row():
                score_out = gr.Textbox(
                    label="Similarity score",
                    interactive=False,
                )
                brand_out = gr.Textbox(
                    label="Matched brand",
                    interactive=False,
                )

    detect_btn.click(
        fn=run_detection,
        inputs=[image_input],
        outputs=[verdict_out, message_out, score_out, brand_out],
    )

    gr.HTML(f"""
    <div style="margin-top:20px;padding:16px;background:rgba(255,107,0,0.08);
         border-radius:12px;border:1px solid rgba(255,107,0,0.2)">
      <p style="font-size:12px;color:#888;font-weight:700;margin-bottom:6px">
        {len(database)} BRANDS SUPPORTED
      </p>
      <p style="font-size:12px;color:#aaa;line-height:1.8">{brand_list}</p>
    </div>
    <div style="text-align:center;margin-top:14px;font-size:12px;color:#555">
      Built by <strong style="color:#ff6b00">Gunasekar</strong> &nbsp;|&nbsp;
      CLIP ViT-B/32 &nbsp;|&nbsp;
      <a href="https://github.com/gunasekar05042005-dot/Fake-Logo-Detector"
         style="color:#ff6b00" target="_blank">GitHub ↗</a>
    </div>
    """)

# ── These 3 lines fix the HF error ────────────
application = app   # Hugging Face looks for this name
handler     = app   # fallback alias
app.launch()        # launches when HF starts the space
