# 🗑️ Smart Waste Classification & Recycling Recommendation System
## Implementation Plan — Step-by-Step Guide

> **Based On:** `prd-smart-waste-classification.md` v1.0  
> **Project Type:** Computer Vision / AI-ML Application  
> **Target Accuracy:** ≥90% top-1 | ≥95% top-3  
> **Inference Latency:** ≤2s on mid-range mobile devices  

---

## 📋 Table of Contents

1. [Project Initialization & Setup](#phase-0-project-initialization--setup)
2. [Phase 1 — Dataset Collection & Baseline Model](#phase-1-dataset-collection--baseline-model-weeks-13)
3. [Phase 2 — Model Optimization & Recommendation Engine](#phase-2-model-optimization--recommendation-engine-weeks-46)
4. [Phase 3 — Mobile/Web App MVP](#phase-3-mobileweb-app-mvp-weeks-79)
5. [Phase 4 — Pilot Testing & Feedback Loop](#phase-4-pilot-testing--feedback-loop-weeks-1012)
6. [Phase 5 — Stretch Features (Post-MVP)](#phase-5-stretch-features-post-mvp)
7. [Tech Stack Summary](#-tech-stack-summary)
8. [Folder Structure](#-folder-structure)
9. [Acceptance Criteria Checklist](#-acceptance-criteria-checklist)

---

## Phase 0: Project Initialization & Setup

> **Goal:** Establish the repository, tooling, environments, and team conventions before any code is written.

### Step 0.1 — Repository & Version Control
- [ ] Create a **GitHub/GitLab repository**: `smart-waste-classifier`
- [ ] Define branching strategy: `main` → `develop` → `feature/*`, `experiment/*`
- [ ] Add `.gitignore` for Python, Node.js, and large model files (`.h5`, `.tflite`, `.onnx`)
- [ ] Create a **GitHub Project Board** with columns: `Backlog`, `In Progress`, `Review`, `Done`

### Step 0.2 — Answer Open Questions (from PRD §11)
Before writing a single line of code, resolve these architectural decisions:

| Question | Decision Needed |
|---|---|
| Which base dataset? | TrashNet / TACO / custom mix |
| Target platform first? | Mobile app, web app, or smart-bin? |
| Inference strategy? | On-device (TFLite/ONNX) or Cloud API? |
| Initial target region? | Geography for recycling rules |

> [!IMPORTANT]
> These decisions directly impact model choice, export format, API design, and front-end tech stack. Resolve them in a kickoff meeting before Phase 1 begins.

### Step 0.3 — Environment Setup
```bash
# Python environment
python -m venv .venv
pip install torch torchvision timm tensorflow opencv-python pillow albumentations scikit-learn matplotlib seaborn pandas tqdm

# Project scaffolding
mkdir -p data/{raw,processed,augmented} models/{checkpoints,exports} src/{data,model,api,app} notebooks tests docs
```

### Step 0.4 — Define Class Taxonomy
Finalize the **8 waste categories** (as per PRD §3.1):

| ID | Category | Examples |
|---|---|---|
| 0 | Plastic | PET bottles, bags, containers |
| 1 | Paper/Cardboard | Newspaper, cardboard boxes |
| 2 | Glass | Bottles, jars |
| 3 | Metal | Cans, foil, scrap |
| 4 | Organic/Biodegradable | Food scraps, garden waste |
| 5 | E-waste | Phones, cables, batteries |
| 6 | Hazardous | Paints, chemicals, syringes |
| 7 | Other/Non-recyclable | Styrofoam, diapers, composite items |

---

## Phase 1: Dataset Collection & Baseline Model (Weeks 1–3)

> **Goal:** Collect and prepare a high-quality, balanced dataset and train a baseline CNN model.

### Step 1.1 — Dataset Sourcing

#### Primary Public Datasets
- [ ] **TrashNet** (~2,500 images, 6 classes) — [garythung/trashnet](https://github.com/garythung/trashnet)
- [ ] **TACO** (Trash Annotations in Context, ~1,500 images with COCO-format annotations) — [tacodataset.org](http://tacodataset.org)
- [ ] **Garbage Classification (Kaggle)** (~15,000 images, 12 classes)

#### Custom Data Collection (if needed)
- [ ] Identify gaps in public datasets (e-waste, hazardous, region-specific items)
- [ ] Collect 200–500 additional images per underrepresented class using smartphones
- [ ] Use **LabelImg** or **Roboflow** for annotation
- [ ] Apply a **data contribution form** if crowdsourcing images

### Step 1.2 — Data Preprocessing Pipeline

```
src/data/
├── dataset.py          # Custom PyTorch Dataset class
├── preprocess.py       # Resize, normalize, de-duplicate
├── augment.py          # Albumentations augmentation pipeline
└── split.py            # Stratified train/val/test split (70/15/15)
```

**Preprocessing Steps:**
- [ ] Resize all images to `224×224` (standard CNN input)
- [ ] Normalize with ImageNet mean/std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- [ ] Remove duplicates using perceptual hashing (`imagehash` library)
- [ ] Stratified split: **70% train / 15% val / 15% test**
- [ ] Generate class distribution chart and ensure no class < 200 samples

### Step 1.3 — Data Augmentation Pipeline

```python
# augment.py — Albumentations pipeline
import albumentations as A

train_transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomShadow(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Augmentation Goals (per PRD §7):**
- Rotation, lighting variation, occlusion, background variation

### Step 1.4 — Baseline Model Training

**Model Architecture:** Transfer learning on **MobileNetV3-Large** (lightweight, mobile-friendly)

```
src/model/
├── model.py            # Model definition (transfer learning)
├── train.py            # Training loop with early stopping
├── evaluate.py         # Accuracy, confusion matrix, per-class metrics
└── config.yaml         # Hyperparameters
```

**Training Configuration:**
```yaml
# config.yaml
model: mobilenet_v3_large
pretrained: true
num_classes: 8
epochs: 50
batch_size: 32
learning_rate: 0.001
optimizer: AdamW
scheduler: CosineAnnealingLR
early_stopping_patience: 7
```

**Training Steps:**
- [ ] Freeze backbone, train classification head (5 epochs)
- [ ] Unfreeze top layers for fine-tuning (remaining epochs)
- [ ] Log metrics with **Weights & Biases (wandb)** or **TensorBoard**
- [ ] Save best checkpoint based on validation accuracy

### Step 1.5 — Baseline Evaluation

- [ ] Compute **top-1** and **top-3** accuracy on the test set
- [ ] Generate **confusion matrix** to identify misclassification pairs
- [ ] Compute **per-class precision, recall, F1-score**
- [ ] Target: Top-1 ≥ 85% for baseline (≥90% after Phase 2 optimization)

**Phase 1 Deliverables:**
- ✅ Cleaned, balanced, augmented dataset (`data/processed/`)
- ✅ Trained baseline model checkpoint (`models/checkpoints/baseline_v1.pth`)
- ✅ Evaluation notebook with confusion matrix and metrics
- ✅ Data pipeline scripts in `src/data/`

---

## Phase 2: Model Optimization & Recommendation Engine (Weeks 4–6)

> **Goal:** Push accuracy to ≥90%, optimize for deployment, and build the recommendation engine.

### Step 2.1 — Model Architecture Experiments

Run experiments with heavier architectures if baseline < 90%:

| Model | Params | Mobile-Friendly | Notes |
|---|---|---|---|
| MobileNetV3-Large | ~5.4M | ✅ Yes | Baseline |
| EfficientNet-B2 | ~9.1M | ✅ Yes | Better accuracy |
| EfficientNet-B4 | ~19M | ⚠️ Moderate | High accuracy |
| ResNet-50 | ~25M | ❌ Heavy | Cloud inference only |

- [ ] Run **hyperparameter sweep** (learning rate, weight decay, batch size) using Optuna or W&B Sweeps
- [ ] Apply **class-weighted loss** to handle any remaining imbalance
- [ ] Experiment with **label smoothing** (ε=0.1) to prevent overconfidence

### Step 2.2 — Model Export & Quantization

For on-device deployment (PRD §5 — Portability):

```bash
# Export to TensorFlow Lite
python src/model/export.py --format tflite --quantize int8 --output models/exports/waste_classifier.tflite

# Export to ONNX
python src/model/export.py --format onnx --output models/exports/waste_classifier.onnx
```

- [ ] **Post-Training Quantization** (INT8) — reduces model size by ~4x with minimal accuracy loss
- [ ] Benchmark latency on a mid-range Android device (target ≤2s)
- [ ] Validate quantized model accuracy (should not drop >2% from full-precision)

### Step 2.3 — Confidence & Fallback Logic

```python
# inference.py
CONFIDENCE_THRESHOLD = 0.65

def classify(image) -> dict:
    probs = model(preprocess(image))
    top1_conf, top1_class = probs.max(dim=1)
    
    if top1_conf < CONFIDENCE_THRESHOLD:
        return {"status": "low_confidence", "prompt": "Please retake photo or select category manually"}
    
    return {
        "category": CLASS_NAMES[top1_class],
        "confidence": round(top1_conf.item(), 3),
        "top3": get_top3(probs),
        "recommendation": RECOMMENDATION_DB[CLASS_NAMES[top1_class]]
    }
```

### Step 2.4 — Recommendation Engine

Build a **rule-based mapping layer** (PRD §3.1 — FR-4):

```
src/recommendation/
├── rules.json              # Category → disposal instructions
├── regions/
│   ├── default.json        # Generic global rules
│   ├── india.json          # India-specific rules
│   └── us.json             # US-specific rules
└── engine.py               # Lookup + region override logic
```

**`rules.json` structure:**
```json
{
  "plastic": {
    "general_instruction": "Rinse and place in the blue recycling bin.",
    "sub_types": {
      "PET bottle": "Crush to save space. Remove cap. Blue bin.",
      "plastic bag": "Do NOT put in blue bin. Take to store drop-off."
    },
    "icon": "♻️",
    "color_code": "#2196F3"
  },
  "e-waste": {
    "general_instruction": "Do not place in any household bin.",
    "sub_types": {},
    "icon": "⚡",
    "color_code": "#FF9800",
    "special_note": "Take to nearest e-waste collection center."
  }
}
```

### Step 2.5 — Cloud Inference API (Optional / Parallel Track)

If cloud inference is chosen (PRD §5 — Scalability: 1,000 concurrent requests):

```
src/api/
├── main.py             # FastAPI application
├── routes/
│   └── classify.py     # POST /classify endpoint
├── schemas.py          # Request/Response Pydantic models
└── Dockerfile
```

**API Contract:**
```
POST /api/v1/classify
Content-Type: multipart/form-data
Body: { image: <file>, region: "IN" }

Response 200:
{
  "category": "plastic",
  "confidence": 0.94,
  "recommendation": "Rinse and place in the blue recycling bin.",
  "top3": [{"label": "plastic", "score": 0.94}, ...],
  "latency_ms": 312
}
```

**Phase 2 Deliverables:**
- ✅ Optimized model with ≥90% top-1 accuracy
- ✅ TFLite + ONNX exports in `models/exports/`
- ✅ `rules.json` recommendation database
- ✅ FastAPI inference server (if cloud path chosen)
- ✅ Confidence/fallback inference pipeline

---

## Phase 3: Mobile/Web App MVP (Weeks 7–9)

> **Goal:** Build a user-facing interface with camera capture, classification results, and scan history.

### Step 3.1 — Platform Decision & Scaffolding

**Recommended MVP Path:** React-based **Web App** (fastest to ship, cross-platform)  
**Future:** React Native mobile app using the same API

```bash
# Web App (Vite + React)
npx create-vite@latest frontend --template react
cd frontend && npm install axios react-router-dom

# OR: Flutter for cross-platform mobile
flutter create waste_classifier_app
```

### Step 3.2 — Core UI Screens

```
frontend/src/
├── pages/
│   ├── Home.jsx            # Landing page + quick scan CTA
│   ├── Scanner.jsx         # Camera capture / file upload
│   ├── Result.jsx          # Classification result + recommendation
│   ├── History.jsx         # Past scan log
│   └── About.jsx           # How it works / FAQ
├── components/
│   ├── Camera.jsx          # Camera API wrapper
│   ├── ResultCard.jsx      # Category badge + confidence bar
│   ├── RecommendationBox.jsx   # Disposal instructions
│   └── FeedbackModal.jsx   # Misclassification correction (FR-5)
└── services/
    └── api.js              # Axios wrapper for /classify endpoint
```

### Step 3.3 — Camera & Upload Feature (FR-1)

```jsx
// Scanner.jsx — Camera capture + file upload
const handleCapture = async (imageBlob) => {
  const formData = new FormData();
  formData.append('image', imageBlob);
  formData.append('region', userRegion);
  const result = await classifyImage(formData);
  navigate('/result', { state: result });
};
```

- [ ] Implement browser `getUserMedia` API for live camera capture
- [ ] Support file upload via `<input type="file" accept="image/*">`
- [ ] Add image preview before submission
- [ ] Show loading spinner during inference call

### Step 3.4 — Results Display (FR-3 & FR-4)

**Result Card UI Elements:**
- Category icon + color-coded badge (e.g., blue for plastic, green for organic)
- Confidence score with visual progress bar
- Disposal recommendation text
- "Was this correct?" feedback button (triggers FR-5 correction flow)
- "Share" and "Scan Again" CTAs

### Step 3.5 — Scan History & Local Storage (FR-6)

```javascript
// history.js — localStorage-based scan log
const saveScan = (result) => {
  const history = JSON.parse(localStorage.getItem('scanHistory') || '[]');
  history.unshift({ ...result, timestamp: new Date().toISOString() });
  localStorage.setItem('scanHistory', JSON.stringify(history.slice(0, 100)));
};
```

- [ ] Store up to 100 scan entries in `localStorage`
- [ ] Display scan history with date, category, thumbnail, confidence
- [ ] Allow deletion of individual or all history entries

### Step 3.6 — Misclassification Feedback (FR-5)

- [ ] "Correct category" dropdown in FeedbackModal
- [ ] Submit correction to backend endpoint: `POST /api/v1/feedback`
- [ ] Backend logs correction with original image hash and correct label for retraining

### Step 3.7 — Offline / On-Device Fallback (FR-7)

- [ ] Bundle `waste_classifier.tflite` or ONNX model via **TensorFlow.js** for browser inference
- [ ] Service Worker for PWA caching of assets
- [ ] Detect offline state and switch to on-device inference automatically

**Phase 3 Deliverables:**
- ✅ Fully functional web app MVP (`frontend/`)
- ✅ Camera capture + file upload working
- ✅ Result display with confidence and recommendation
- ✅ Scan history (localStorage)
- ✅ Misclassification feedback form
- ✅ PWA with offline fallback

---

## Phase 4: Pilot Testing & Feedback Loop (Weeks 10–12)

> **Goal:** Deploy to a pilot group, collect real-world data, retrain model, and validate success metrics.

### Step 4.1 — Deployment

```bash
# Backend API — Docker
docker build -t waste-classifier-api .
docker run -p 8000:8000 waste-classifier-api

# Cloud options
# - Google Cloud Run (recommended for auto-scaling to 1,000 concurrent)
# - AWS Lambda + API Gateway
# - Railway / Render
```

- [ ] Deploy FastAPI backend on **Google Cloud Run** or **Railway**
- [ ] Deploy frontend on **Vercel** or **Netlify**
- [ ] Set up **uptime monitoring** (UptimeRobot) — target 99.5% availability (PRD §5)
- [ ] Configure HTTPS, CORS, and rate limiting

### Step 4.2 — Pilot Group Rollout

- [ ] Recruit 20–50 pilot users (household, campus, or educational context)
- [ ] Set up **feedback collection pipeline** (corrections → retraining queue)
- [ ] Track: classification accuracy, user correction rate, latency, session count

### Step 4.3 — Monitoring & Analytics

```
monitoring/
├── metrics_dashboard.py    # Prometheus + Grafana OR built-in Cloud Run metrics
├── error_tracker.py        # Sentry integration for runtime errors
└── feedback_logger.py      # Correction log ingestion
```

**Key Metrics to Track (PRD §8):**

| Metric | Target | Monitoring Tool |
|---|---|---|
| Top-1 Classification Accuracy | ≥90% | Custom eval script |
| Avg. Inference Latency | ≤2s | Cloud Run logs |
| User Correction Rate | <10% of scans | Feedback DB query |
| Weekly Active Users | As defined per context | Analytics |
| Recycling contamination reduction | ≥15% (pilot site) | Pilot partner data |

### Step 4.4 — Retraining Pipeline

```
src/retraining/
├── collect_corrections.py  # Pull corrections from feedback DB
├── retrain.py              # Incremental fine-tuning on new data
├── validate.py             # Ensure accuracy doesn't regress
└── deploy_model.py         # OTA model versioning (FR-8)
```

**Retraining Trigger:** Weekly batch job or when correction pool reaches 500 new samples

- [ ] Validate retrained model does not regress (accuracy must not drop >1%)
- [ ] Push new `.tflite`/`.onnx` to CDN for OTA update (FR-8)
- [ ] Version all models (e.g., `waste_classifier_v1.2.tflite`)

### Step 4.5 — Accessibility Audit (PRD §5)

- [ ] Run **Lighthouse** accessibility audit (target score ≥90)
- [ ] Verify WCAG 2.1 AA color contrast ratios
- [ ] Add `aria-label` attributes to all interactive elements
- [ ] Test with screen reader (NVDA / VoiceOver)

**Phase 4 Deliverables:**
- ✅ Production deployment live
- ✅ Pilot group active and submitting feedback
- ✅ Model accuracy ≥90% confirmed on real-world data
- ✅ Retraining pipeline operational
- ✅ All PRD success metrics tracked on dashboard

---

## Phase 5: Stretch Features (Post-MVP)

> **Goal:** Enhance the system with advanced capabilities based on pilot learnings.

### Step 5.1 — Location-Aware Guidance (PRD §3.1)
- [ ] Add geolocation API to detect user's region
- [ ] Serve region-specific `rules.json` from backend (e.g., `/rules?region=IN-MH`)
- [ ] Admin UI to update regional rules without a deployment

### Step 5.2 — Multi-Object Detection
- [ ] Switch from classification to **object detection** (YOLOv8 / EfficientDet)
- [ ] Draw bounding boxes around each waste item in a single photo
- [ ] Return classification + recommendation per detected item

### Step 5.3 — Gamification
- [ ] User accounts with recycling streak tracking
- [ ] Points per correct disposal action
- [ ] Leaderboard (campus/housing society level)
- [ ] Badges (e.g., "Plastic Hero", "Zero Waste Week")

### Step 5.4 — Admin Analytics Dashboard
- [ ] Waste composition breakdown by category (pie/bar charts)
- [ ] Scan volume trends over time
- [ ] Per-region correction rates to flag model weaknesses
- [ ] Export reports as CSV/PDF

### Step 5.5 — Barcode / Label OCR Fusion
- [ ] Integrate **Google ML Kit** barcode scanner
- [ ] Fetch product type from Open Food Facts API
- [ ] Fuse barcode result with visual classification for higher confidence

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| **ML Framework** | PyTorch (training) + TensorFlow Lite / ONNX (deployment) |
| **Model Architecture** | MobileNetV3 / EfficientNet (transfer learning) |
| **Data Augmentation** | Albumentations |
| **Experiment Tracking** | Weights & Biases (W&B) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React (Vite) + Axios |
| **Database** | PostgreSQL (feedback log) + Redis (caching) |
| **Containerization** | Docker + Docker Compose |
| **Cloud Deployment** | Google Cloud Run / Vercel |
| **Monitoring** | Sentry + UptimeRobot |
| **CI/CD** | GitHub Actions |
| **Model Serving (on-device)** | TensorFlow.js (web) / TFLite (mobile) |

---

## Folder Structure

```
smart-waste-classifier/
├── data/
│   ├── raw/                    # Original downloaded datasets
│   ├── processed/              # Cleaned, split datasets
│   └── augmented/              # Augmented training data
├── models/
│   ├── checkpoints/            # .pth training checkpoints
│   └── exports/                # .tflite, .onnx, .pb exports
├── src/
│   ├── data/                   # Dataset, preprocess, augment, split scripts
│   ├── model/                  # Model definition, train, evaluate, export
│   ├── recommendation/         # Rules engine + regional JSON files
│   ├── api/                    # FastAPI server
│   └── retraining/             # Feedback-driven retraining pipeline
├── frontend/                   # React web app
├── notebooks/                  # Jupyter EDA, training experiments
├── monitoring/                 # Metrics, alerts, dashboards
├── tests/                      # Unit + integration tests
├── docs/                       # Architecture diagrams, API docs
├── .github/workflows/          # CI/CD pipelines
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Acceptance Criteria Checklist

### Model
- [ ] Top-1 accuracy ≥90% on held-out test set
- [ ] Top-3 accuracy ≥95% on held-out test set
- [ ] Average inference latency ≤2s on mid-range device
- [ ] Model exportable to TFLite (INT8) and ONNX
- [ ] Quantized model accuracy drop ≤2%

### Application
- [ ] Camera capture and file upload both functional (FR-1)
- [ ] Classification result returned within 2s (FR-2)
- [ ] Confidence score displayed (FR-3)
- [ ] Disposal recommendation shown for all 8 categories (FR-4)
- [ ] Misclassification correction flow works end-to-end (FR-5)
- [ ] Scan history persisted locally (FR-6)
- [ ] App works with no internet connection (FR-7)
- [ ] Model OTA update mechanism implemented (FR-8)

### Operations
- [ ] Cloud API uptime ≥99.5% over pilot period
- [ ] Backend handles ≥1,000 concurrent requests
- [ ] Lighthouse accessibility score ≥90
- [ ] No PII stored with image data by default
- [ ] Retraining pipeline tested with 500 correction samples

---

## Timeline Overview

```
Week  1-3  │ Phase 1 │ Dataset collection, preprocessing, baseline training
Week  4-6  │ Phase 2 │ Model optimization, export, recommendation engine, API
Week  7-9  │ Phase 3 │ Web app MVP — scanner, results, history, feedback
Week 10-12 │ Phase 4 │ Deployment, pilot testing, retraining pipeline, monitoring
Post-MVP   │ Phase 5 │ Location-aware rules, multi-object detection, gamification
```

---

*Generated from `prd-smart-waste-classification.md` v1.0 | Last updated: October 2026*
