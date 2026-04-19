# Fake Logo Detection System 🔍

An AI-powered logo authenticity verification system built with Python and CLIP (ViT-B/32). Upload any brand logo and instantly detect whether it is authentic or fake using deep learning embeddings and cosine similarity matching.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-green?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Vercel](https://img.shields.io/badge/Deployed-Vercel-black?style=flat-square&logo=vercel)

---

## What It Does

This system takes a logo image as input and tells you:
- Whether the logo is **AUTHENTIC** or **FAKE**
- The **confidence level** (High / Medium / Low)
- The **similarity score** (0.0 to 1.0)
- The **closest matching brand** in the database

---

## How It Works

1. Authentic brand logos are embedded into 512-dimensional vectors using **OpenAI CLIP ViT-B/32**
2. When a logo is submitted, its embedding is extracted using the same model
3. **Cosine similarity** is computed against all stored authentic brand embeddings
4. Score ≥ 0.80 → **AUTHENTIC** — Score < 0.80 → **FAKE**

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Model | CLIP ViT-B/32 (OpenAI) via open-clip-torch |
| Deep Learning | PyTorch |
| Backend API | Flask + Flask-CORS |
| Security | Flask-Limiter, API Key Auth, File Validation |
| Image Processing | Pillow |
| Similarity Search | scikit-learn cosine similarity |
| Deployment | Vercel |

---

## Features

- Detects fake and counterfeit logos with high accuracy
- Supports 29 major brands out of the box
- REST API with 5 endpoints
- Rate limiting — 10 requests per minute per IP
- API key authentication for admin routes
- File type validation using magic bytes
- EXIF metadata stripping for privacy
- Request logging to file
- Lightweight fallback mode for Vercel deployment

---

## Supported Brands

Nike, Adidas, Apple, Samsung, Sony, Google, Microsoft, Amazon, Coca-Cola, Pepsi, McDonald's, Starbucks, Louis Vuitton, Gucci, Puma, Ferrari, BMW, Mercedes, Toyota, Tesla, Intel, NVIDIA, Meta, YouTube, Netflix, Spotify, Disney, Visa, Mastercard

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/` | None | API info and endpoints |
| GET | `/health` | None | Health check |
| POST | `/detect` | None | Detect logo (upload image) |
| GET | `/brands` | None | List all brands in database |
| POST | `/add-brand` | API Key | Add new authentic brand |
| DELETE | `/brand/<name>` | API Key | Remove a brand |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-logo-detector.git
cd fake-logo-detector

# Install dependencies
pip install open-clip-torch Pillow scikit-learn flask flask-cors flask-limiter python-dotenv tqdm

# Create .env file
cp .env.template .env
# Edit .env and set your API_KEY

# Add authentic logos
# Create folders: authentic_logos/<brand_name>/logo.png

# Build the database
python fake_logo_detector_allinone_1.py --build

# Run locally
python app.py
```

---

## Usage

```bash
# Build logo database
python fake_logo_detector_allinone_1.py --build

# Detect a logo
python fake_logo_detector_allinone_1.py --detect suspect_logo.png

# Compare two logos directly
python fake_logo_detector_allinone_1.py --compare logo_a.png logo_b.png

# List all brands in database
python fake_logo_detector_allinone_1.py --list

# Start REST API server
python fake_logo_detector_allinone_1.py --api
```

---

## API Usage Example

```bash
# Detect a logo via API
curl -X POST https://your-app.vercel.app/detect \
     -F "image=@suspect_logo.png"

# Response
{
  "success": true,
  "result": {
    "verdict": "AUTHENTIC",
    "confidence": "HIGH",
    "similarity_score": 0.9341,
    "matched_brand": "nike",
    "message": "Very close CLIP match to 'nike'."
  }
}

# Add a new brand (requires API key)
curl -X POST https://your-app.vercel.app/add-brand \
     -H "X-API-Key: your-secret-key" \
     -F "brand_name=nike" \
     -F "image=@nike_logo.png"
```

---

## Project Structure

```
fake-logo-detector/
├── app.py                          # Flask REST API (Vercel entry point)
├── fake_logo_detector_allinone_1.py  # All-in-one detector + CLI
├── vercel.json                     # Vercel deployment config
├── requirements.txt                # Python dependencies
├── .env.template                   # Environment variables template
├── .gitignore                      # Git ignore rules
├── authentic_logos/                # Brand logo images (one folder per brand)
│   ├── nike/
│   │   └── logo.png
│   └── adidas/
│       └── logo.png
└── logo_embeddings_clip.pkl        # Auto-generated brand database
```

---

## Detection Results Explained

| Score | Verdict | Meaning |
|---|---|---|
| 0.88 – 1.00 | AUTHENTIC (HIGH) | Very close match to authentic brand |
| 0.80 – 0.87 | AUTHENTIC (MEDIUM) | Matches brand with moderate confidence |
| 0.60 – 0.79 | FAKE (MEDIUM) | Resembles brand but too different |
| 0.00 – 0.59 | FAKE (HIGH) | No match found — definitely fake |

---

## Author

**Gunasekar**
BCA Student — Ranipet, Tamil Nadu, India
OCI 2025 Certified AI Foundations Associate

---

## License

This project is open source and available under the [MIT License](LICENSE).
