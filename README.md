# Transformer Grammar Correction

A production-quality, ground-up implementation of the Transformer architecture (Vaswani et al., 2017) designed for **Grammar Error Correction (GEC)**. This project is optimized for performance on modern GPUs (H100/A100) using **Mixed Precision (AMP)** and follows strict engineering standards.

## 🚀 Features

*   **Custom Transformer:** Built from scratch in PyTorch (Encoder-Decoder).
*   **Production Ready:** 
    *   Strict **Google-style** code structure (`src` layout).
    *   Fully type-hinted and documented.
    *   Structured logging and checkpoint management.
*   **H100 Optimized:**
    *   **Mixed Precision (AMP):** ~2x speedup and 40% less VRAM.
    *   **Resume Capability:** Robust training that recovers from crashes/disconnects (critical for Colab).
*   **Dataset:** Integration with `liweili/c4_200m` (cleaned C4 dataset) for massive-scale pretraining.

## 🛠️ Installation

 Clone the repository and install in **editable mode**:

```bash
git clone <your-repo-url>
cd grammar_correction
pip install -e .
```

*Note: Ensure you have PyTorch installed with CUDA support.*

## ⚡ Usage

### 1. Training
Run the trainer module directly. 

**Quick Test (CPU/Local):**
```bash
python -m grammar_correction.trainer --num_samples 100 --epochs 1 --batch_size 4
```

**Production Run (Google Colab / H100):**
```bash
python -m grammar_correction.trainer \
    --batch_size 256 \
    --epochs 5 \
    --d_model 512 \
    --resume_from model_epoch_0.pt
```

### 2. Inference
Use the inference module to correct sentences:

```bash
python -m grammar_correction.inference "model_epoch_5.pt" "He go to school yesterday."
```
*Output: "He went to school yesterday."*

## 📂 Project Structure

```text
grammar_correction/
├── pyproject.toml              # Dependencies & Build Config
├── README.md                   # Documentation
├── src/
│   └── grammar_correction/     # Main Package
│       ├── trainer.py          # Training Loop (AMP enabled)
│       ├── model.py            # Transformer Architecture
│       ├── optimizer.py        # Noam Learning Rate Scheduler
│       ├── dataset.py          # HuggingFace Dataset Wrapper
│       └── ...
└── tests/                      # Unit Tests (pytest)
```

## 🧪 Testing
Run the test suite to verify architecture and data loading:

```bash
pytest tests/
```

## 📜 License
MIT
