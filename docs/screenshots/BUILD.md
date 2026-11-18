# Build the screenshots represent

The screenshots in this folder were captured on **2026-11-18** from a clean evaluator setup, not from a developer
machine's existing data.

| Item | Value |
|---|---|
| Application source | `devrajsengar786-sssup/brain-tumor-detection`, `main` at commit **`b8eea7a`** (merge of PR #1). `app.py`, `inference.py` and `retrieval/gallery2d.py` are unchanged by the documentation commit that adds these files |
| Checkout | fresh `git clone` of the public repository |
| Python | 3.12.9, 64-bit, Windows 11 |
| Packages | `pip install -r requirements.txt` into a new virtual environment; exact versions in `requirements-lock.txt` (TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.1.3, Streamlit 1.67.0, nibabel 5.3.2) |
| Model weights | downloaded into an empty Hugging Face cache from `mvdu/brain-tumor-models`: `binary_resnet50.h5` (sha256 `0bcc00e6be9dc6ea…`), `multiclass_resnet50.h5` (sha256 `3397f4968778f58a…`) |
| Input bundle | rebuilt from the public UPENN-GBM mirror with `python -m retrieval.cli build-upenn-bundle` (147 complete cases, 147 validated, split 103 / 22 / 22, seed 42); its `gallery_metadata.csv`, `demo_metadata.csv` and `evaluation/secondary_labels.csv` were identical to the project's Colab-built bundle |
| Reference gallery | `python -m retrieval.gallery2d prepare`: 125 slices exported and verified, 103 indexed, 22 held out; identical slice pixels and embeddings (cosine 1.000000) to the project's earlier gallery |
| Query shown | held-out demo scan **`sub-011`** (*Try a demo scan*), never part of the reference library |
| Browser | Microsoft Edge 157 (headless, driven by Playwright); desktop 1440 × 900, phone 390 × 844 at 2× |

## What each screenshot shows

| File | State | Verified on capture |
|---|---|---|
| `01-empty-state.png` | Start screen | Upload panel, *Try a demo scan*, six held-out sample scans |
| `02-full-result-page.png` | Whole results page | Every section below on one page |
| `03-analysis-gradcam.png` | Scan + Results panels | *Tumour detected*, model confidence 89.0%, tumour score 89.0% vs 50% threshold, predicted type Meningioma 65.8% with all four class probabilities, Grad-CAM overlay with colour scale |
| `04-similar-cases.png` | Similar cases | Five ranked references, 2D visual similarity 0.81, 0.81, 0.80, 0.79, 0.79; glioblastoma-only scope note |
| `05-comparison.png` | Comparison | Uploaded scan (model outputs) vs reference rank 1 (`UPENN-GBM-00135_11`: glioblastoma, 56 years, male, T1-Gd axial slice 93, 19.0 mL, IDH1 wildtype, MGMT methylated) with TCIA citation |
| `06-side-by-side.png` | Grad-CAM side by side | Original and overlay viewers |
| `07-invalid-upload.png` | Text file uploaded as `.png` | "This file could not be read as a JPG or PNG image."; earlier results cleared |
| `08-mobile-result-top.png` | Phone layout | No horizontal overflow (checked on the empty and the results page) |
| `10-missing-gallery.png` | App started with `GALLERY_2D_DIR` pointing to a missing folder | *Reference library unavailable* with the rebuild command; prediction still available |

Also checked during the same run: choosing reference rank 3 changed the comparison without re-running the models
(the "Analysed in 4.3 s" label was unchanged).

The numbers in the screenshots are model outputs for one public held-out slice. *Model confidence* is the
classifier's score for its predicted label; *2D visual similarity* is the cosine similarity of ResNet50 image
embeddings. Neither is a measured accuracy or a diagnosis.
