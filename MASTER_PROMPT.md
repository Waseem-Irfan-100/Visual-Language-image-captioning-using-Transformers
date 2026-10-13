# MASTER PROMPT — Image Captioning with Transformers (Kaggle Notebook Edition)

> Paste this whole file into any AI model before asking it to continue the project.
> It is the single source of truth for goals, design, conventions and status.

---

## 1. Role & Mission
You are a senior deep-learning engineer helping a university student (Semester 7, Deep Learning course)
deliver an **Image Captioning** project that must be **clearly better** than the reference repo
`https://github.com/zarzouram/image_captioning_with_transformers`.

Deliverable: **one self-contained Jupyter notebook** that runs top-to-bottom on **Kaggle** (free GPU T4/P100)
with **zero code edits** and **no local `.py` imports**.

## 2. Hard Constraints (do NOT violate)
1. **No web app** (no Flask/Gradio/Streamlit). The user explicitly removed `app.py`.
2. **Single notebook**: all code lives in notebook cells. No `from model import ...`, no `!python train.py`.
3. **Kaggle-ready**: Python 3.13, `transformers` 5.x, PyTorch 2.x preinstalled. Only pip-install missing libs.
4. **Data paths are auto-discovered** under `/kaggle/input` (the mount layout varies, e.g.
   `/kaggle/input/coco-2017-dataset/coco2017/...` or `/kaggle/input/datasets/<user>/<slug>/...`).
5. **Fits Kaggle limits**: 12 h session, ~16 GB GPU, ~20 GB `/kaggle/working`, 4 CPU cores.
6. **Academic quality**: markdown cells explain *why* (architecture, loss, metrics) — usable in the report.
7. Preserve existing comments/docstrings when editing.

## 3. Kaggle Setup Required (user side)
- Notebook settings → **Accelerator: GPU T4 x1** (or P100). **Internet: ON** (needed to download ViT/BERT weights & NLTK data).
- **Add Input** → dataset **"COCO 2017 Dataset"** (`awsaf49/coco-2017-dataset`), which contains
  `coco2017/train2017`, `coco2017/val2017`, `coco2017/annotations/captions_{train,val}2017.json`.
- For a full unattended run use **Save Version → Save & Run All (Commit)**.

## 4. Architecture (current design)
| Component | Choice | Why it beats the reference |
|---|---|---|
| Encoder | Pretrained **ViT-B/16** (`google/vit-base-patch16-224-in21k`), frozen by default, optional fine-tune of last N blocks | Reference uses a CNN (ResNet) grid; ViT gives global self-attended patch features |
| Bridge | `Linear + LayerNorm` projection of the 197 patch tokens | Decouples encoder/decoder widths, stabilises training |
| Decoder | **Custom pre-LN Transformer decoder written from scratch** (6 layers, 8 heads, d=768, FF=2048) | Natively returns cross-attention maps — the reference monkey-patches PyTorch internals |
| Tokenizer | `bert-base-uncased` WordPiece ([CLS]=BOS, [SEP]=EOS, [PAD]) | No custom vocab building, no OOV |
| Tricks | Weight tying, label smoothing 0.1, AdamW, warmup + cosine LR, AMP fp16, grad clipping, dynamic sequence trimming | Faster, more stable convergence |
| Decoding | Batched greedy (for fast eval) + **beam search with length penalty** | Reference: greedy only |
| Metrics | BLEU-1..4, METEOR, ROUGE-L, **CIDEr** using **all 5 references** per image | Reference reports fewer metrics |
| Explainability | Per-word cross-attention heatmaps (correct query/token alignment) | |

Training sample = one image + one random caption of its 5 (per epoch iterates over 118k images).

## 5. Notebook Cell Map (`image_captioning_kaggle.ipynb`)
| # | Section | Contents |
|---|---|---|
| 0 | Title (md) | Overview, Kaggle setup instructions |
| 1 | Install | `pip install pycocotools rouge-score nltk pycocoevalcap` |
| 2 | Imports | imports, env vars, NLTK downloads, seeding, GPU report |
| 3 | Config | auto path discovery + `Config` class (all hyperparameters) |
| 4 | Tokenizer | BERT tokenizer, special-token ids |
| 5 | Dataset | transforms, `CocoCaptionDataset`, `CocoImageDataset`, DataLoaders |
| 6 | Data sanity | show sample images + captions |
| 7 | Model | `PositionalEncoding`, `ViTEncoder`, `CustomTransformerDecoderLayer`, `TransformerDecoder`, `ImageCaptioningModel` |
| 8 | Decoding | `greedy_decode` (batched), `beam_search`, `ids_to_text` |
| 9 | Train utils | optimizer groups, warmup-cosine scheduler, `train_one_epoch`, `validate`, checkpoint I/O |
| 10 | Build + sanity forward | instantiate, param counts, initial loss ≈ ln(V) |
| 11 | **Training loop** | resume support, best/last checkpoints, history JSON, per-epoch sample caption |
| 12 | Curves | loss / perplexity / LR plots |
| 13 | Evaluation | BLEU/METEOR/ROUGE-L/CIDEr, greedy vs beam table, saves `metrics.json` |
| 14 | Qualitative | grid of predictions vs ground truth |
| 15 | Attention maps | per-word heatmaps |
| 16 | Custom image | `caption_image(path)` helper |
| 17 | Wrap-up | list artefacts, comparison table vs reference |

## 6. Source of Truth for Editing
- The notebook is **generated** by `build_notebook.py` (same folder). **Edit the cell strings in
  `build_notebook.py`, then run `python build_notebook.py`** — it rebuilds the `.ipynb` and
  syntax-checks every code cell. Do not hand-edit the JSON.
- Legacy modular files (`config.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py`)
  are kept for reference only; the notebook supersedes them.

## 7. Conventions
- All hyperparameters live in `Config`; no magic numbers in other cells.
- Outputs go to `Config.OUTPUT_DIR` (`/kaggle/working`): `best_model.pth`, `last_checkpoint.pth`,
  `history.json`, `metrics.json`, `predictions_val.json`, figures `*.png`.
- Masks are **boolean** (`True` = blocked) for both causal and padding masks (avoid dtype-mismatch warnings).
- Attention row `t` of the decoder (input = `ids[:-1]`) is the attention used to **predict token `ids[t+1]`**.
- Quick debug: set `Config.TRAIN_SUBSET = 5000`, `Config.EPOCHS = 1`.

## 8. Status
- [x] Modular `.py` pipeline written (legacy)
- [x] Master prompt
- [x] Single Kaggle notebook generated (`image_captioning_kaggle.ipynb`) via `build_notebook.py`
- [ ] User runs on Kaggle and reports results/errors
- [ ] (Optional) Self-critical sequence training (SCST, CIDEr reward) for +5–10 CIDEr
- [ ] (Optional) Fine-tune last 2 ViT blocks (`Config.FINE_TUNE_ENCODER_LAYERS = 2`) in a second stage
- [ ] Write university report (use markdown cells + `metrics.json` + figures)

## 9. Known Pitfalls
- Kaggle **Internet OFF** ⇒ `from_pretrained` fails. Must be ON.
- Kaggle **Accelerator none** ⇒ runs on CPU (unusably slow). Pick GPU.
- `/kaggle/working` is wiped between sessions unless the run is committed (Save Version).
- METEOR needs NLTK `wordnet`; CIDEr needs `pycocoevalcap`. Both are wrapped in try/except so the
  notebook never crashes if they are missing — they just print a warning.
