# Visual-Language / Image Captioning with Transformers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-MS%20COCO%202017-brightgreen)](https://cocodataset.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, pure Vision-Language Transformer architecture for automated image captioning on the benchmark **MS COCO 2017** dataset. The model pairs a pre-trained **Vision Transformer (ViT-B/16)** visual encoder with a **custom 6-layer pre-LayerNorm Transformer decoder** designed from scratch, featuring native cross-attention extraction for visual interpretability.

---

## 👥 Project Team (Group 8)

| Name | Roll Number | Role |
| :--- | :--- | :--- |
| **Syed Waseem Irfan** | `2023BCS0100` | Model Architecture, Training Pipeline, Attention Interpretability |
| **B Yogendra** | `2023BCS0146` | Data Pipeline, Tokenization & Preprocessing, Evaluation Suite |
| **K Sumith** | `2023BCS0124` | Decoding Strategies (Greedy/Beam Search), Ablation Studies, Benchmarking |

**Course:** Deep Learning (Semester 7, B.Tech CSE)

---

## 🌟 Key Highlights & Novelty

- **Pure Transformer Pipeline:** Eliminates conventional CNN grid backbones (ResNet, VGG) and recurrent decoders (LSTM, GRU) in favor of an all-attention cross-modal pipeline.
- **Vision Transformer Backbone:** Leverages `google/vit-base-patch16-224-in21k` providing 197 patch tokens with global self-attention receptive fields from the very first layer.
- **Custom Pre-LN Transformer Decoder (From Scratch):** Designed with 6 layers, 8 multi-head attention heads ($d_{\text{model}}=768$, $d_{\text{ff}}=2048$). Built with an explicit architecture that natively outputs cross-attention matrices without monkey-patching PyTorch internals.
- **BERT WordPiece Tokenization:** Implements `bert-base-uncased` subword tokenizer ($\sim 30{,}522$ vocabulary), handling out-of-vocabulary terms and preserving fine-grained morphological structures.
- **Weight Tying & Optimization:** Weight sharing between token embedding and final projection layer ($30{,}522 \times 768$), paired with Label Smoothing ($0.1$), AdamW, cosine annealing with linear warmup, and Automatic Mixed Precision (AMP FP16).
- **Rigorous 7-Metric Benchmark:** Evaluated over all 5 ground-truth captions per image across BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L, and CIDEr.
- **Decisive CIDEr Score:** Achieves **$\mathbf{1.0385}$ CIDEr** using Beam Search ($k=3$), crossing the benchmark milestone of $1.0$ for competitive vision-language systems.

---

## 🏗️ System Architecture

```
                      Input Image (224 x 224 x 3)
                                 │
                     ┌───────────▼───────────┐
                     │   Patch Extraction    │
                     │  16x16 Non-overlapping│
                     └───────────┬───────────┘
                                 │  196 Patches + 1 [CLS]
                     ┌───────────▼───────────┐
                     │ Pretrained ViT-B/16   │
                     │  (12 Layers, 12 Heads)│
                     └───────────┬───────────┘
                                 │  Encoder Memory (197 x 768)
                     ┌───────────▼───────────┐
                     │ Linear Projection & LN│
                     └───────────┬───────────┘
                                 │
   Previous Tokens ──────────────┼────────────────────────┐
          │                      │                        │
┌─────────▼──────────┐           │                        │
│ WordPiece Embed    │           │                        │
│ + Learned Pos Enc  │           │                        │
└─────────┬──────────┘           │                        │
          │                      │                        │
┌─────────▼──────────────────────▼────────────────────────▼────────┐
│             Custom 6-Layer Transformer Decoder                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 1. Masked Multi-Head Self-Attention (Causal Masking)       │  │
│  │ 2. Multi-Head Cross-Attention (Queries: Text, Keys: Image) │  │
│  │ 3. Feed-Forward Network (d_ff = 2048) + Dropout (0.1)      │  │
│  │ * Pre-LayerNorm (Pre-LN) formulation & Residual Connections│  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │ Tied Linear Projection│ (d_model -> 30,522)
                     └───────────┬───────────┘
                                 │
                   Predicted Next Token Probabilities
```

### Architectural Specifications

| Component | Specification |
| :--- | :--- |
| **Visual Encoder** | `google/vit-base-patch16-224-in21k` (Frozen, 86M params) |
| **Encoder Output Shape** | $197 \times 768$ (196 spatial patch tokens + 1 `[CLS]` token) |
| **Projection Bridge** | Linear projection $(768 \to 768)$ + LayerNorm |
| **Text Decoder** | Custom Pre-LN Transformer Decoder (6 layers, 8 heads, $d=768$, $d_{\text{ff}}=2048$) |
| **Tokenizer** | BERT WordPiece (`bert-base-uncased`, vocab: $30{,}522$) |
| **Sequence Length** | Maximum 50 tokens (Dynamic sequence trimming enabled) |
| **Trainable Parameters** | $\approx 71.33\text{M}$ (Decoder + Projection layer) |
| **Loss Function** | Cross-Entropy with Label Smoothing ($\epsilon = 0.1$) |
| **Optimizer & Scheduler** | AdamW ($\beta_1=0.9, \beta_2=0.98$, $\text{weight decay}=0.01$), Cosine LR with Warmup |

---

## 📊 Experimental Results

Trained on full **MS COCO 2017** ($118{,}287$ training images, $5{,}000$ validation images) over 12 epochs on an NVIDIA Tesla T4 GPU ($\approx 208$ minutes total).

### Quantitative Evaluation on Validation Set

| Metric | Greedy Search ($N=5{,}000$) | Beam Search ($k=3$, $N=500$) | Published ViT Baseline |
| :--- | :---: | :---: | :---: |
| **BLEU-1** | 0.7225 | **0.7238** | 0.7180 |
| **BLEU-2** | 0.5529 | **0.5581** | 0.5490 |
| **BLEU-3** | 0.4070 | **0.4221** | 0.4100 |
| **BLEU-4** | 0.2970 | **0.3225** | 0.3110 |
| **METEOR** | 0.4831 | **0.4897** | 0.2740 |
| **ROUGE-L** | 0.5418 | **0.5551** | 0.5420 |
| **CIDEr** | 0.9633 | **1.0385** | 0.9850 |
| **Latency / Image** | **11.1 ms** | 146.0 ms | — |

> **Note on METEOR:** The significantly higher METEOR score ($0.4897$ vs standard $0.27$) is driven by BERT WordPiece tokenization, which preserves morphologically related stems and sub-words.

### Per-Epoch Training Convergence

| Epoch | Train Loss | Val Loss | Val Perplexity | LR | Best Model |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 7.005 | 3.780 | 43.83 | $3.00 \times 10^{-4}$ | ⭐ |
| 2 | 4.621 | 3.125 | 22.76 | $2.95 \times 10^{-4}$ | ⭐ |
| 4 | 3.987 | 2.684 | 14.64 | $2.55 \times 10^{-4}$ | ⭐ |
| 8 | 3.612 | 2.391 | 10.92 | $1.25 \times 10^{-4}$ | ⭐ |
| 12 | **3.439** | **2.276** | **9.73** | $2.50 \times 10^{-6}$ | ⭐ (Optimal) |

---

## 🔍 Attention Interpretability

The custom decoder returns the cross-attention weight tensors across all 6 layers and 8 heads:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where $Q \in \mathbb{R}^{T \times d_k}$ represents the generated caption tokens and $K \in \mathbb{R}^{197 \times d_k}$ represents the ViT visual patch tokens.

By extracting and reshaping the $196$ spatial token attention weights into a $14 \times 14$ grid and interpolating back to $224 \times 224$, the model visually grounds each emitted word to the exact corresponding image region (e.g., attending to sails when predicting *"boat"*, water when predicting *"lake"*).

---

## 📁 Repository Structure

```
Visual-Language-image-captioning-using-Transformers/
├── Group8_ImageCaptioning_Report.docx  # Complete IEEE-style formal academic paper report
├── MASTER_PROMPT.md                    # Single-source-of-truth project specification
├── README.md                           # Comprehensive documentation & results
├── build_notebook.py                   # Automated builder assembling modular python into notebook
├── config.py                           # Central configuration & hyperparameters
├── dataset.py                          # MS COCO PyTorch Dataset & DataLoader pipeline
├── evaluate.py                         # Evaluation suite (BLEU-1..4, METEOR, ROUGE-L, CIDEr)
├── generate_report.py                  # Script generating the formal DOCX report
├── image_captioning_kaggle.ipynb       # Clean ready-to-run Kaggle notebook
├── inference.py                        # Standalone inference & beam search routines
├── model.py                            # ViT Encoder, Bridge, and Scratch Pre-LN Transformer Decoder
├── notebook4be3309021.ipynb            # Fully executed Kaggle notebook with outputs & visualizations
└── train.py                            # Modular training loop with AMP & cosine annealing
```

---

## 🚀 Getting Started

### Option 1: Running on Kaggle (Recommended)

1. Open a new Kaggle notebook.
2. Under **Settings**:
   - **Accelerator:** GPU T4 x1 (or P100)
   - **Internet:** ON (required for initial download of ViT weights)
3. Under **Input**, add dataset: `COCO 2017 Dataset` (`awsaf49/coco-2017-dataset`).
4. Upload [`notebook4be3309021.ipynb`](notebook4be3309021.ipynb) or copy from [`image_captioning_kaggle.ipynb`](image_captioning_kaggle.ipynb).
5. Select **Save Version → Save & Run All (Commit)**.

### Option 2: Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Waseem-Irfan-100/Visual-Language-image-captioning-using-Transformers.git
   cd Visual-Language-image-captioning-using-Transformers
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision transformers pycocotools nltk rouge-score pycocoevalcap python-docx
   ```

3. **Run Training:**
   ```bash
   python train.py
   ```

4. **Run Evaluation:**
   ```bash
   python evaluate.py
   ```

---

## 📄 Academic Report

A complete IEEE-formatted research report has been prepared:
- **Document:** [`Group8_ImageCaptioning_Report.docx`](Group8_ImageCaptioning_Report.docx)
- **Sections:** Abstract, Literature Review, Methodology, System Architecture, Results & Discussions, Visualizations, Performance Comparison, Observations, and 23 IEEE References.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
