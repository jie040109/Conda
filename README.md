# Conda

Official PyTorch implementation of the paper:  
[“Conda: Column-Normalized Adam for Training Large Language Models Faster”](https://arxiv.org/abs/2509.24218)

---

## 📥 Installation

```bash
cd Conda
pip install -e .
```

## 🚀 Examples

This repository includes two example training setups using `conda_torch`:

- `examples/gpt2/` — GPT-2 pre-training on Openwebtext  
- `examples/llama/` — LLaMA pre-training on C4

Below are the exact steps to reproduce both examples.

---

## ✅ 1. GPT-2 

### **Step 1 — Install dependencies**
```bash
cd examples/gpt2
conda env create -f environment.yml
conda activate gpt2
```
### **Step 2 — Conda for GPT-2 pre-training**
```bash
# gpt2-125m
bash scripts/train_gpt2_125m_conda.sh
# gpt2-355m
bash scripts/train_gpt2_355m_conda.sh
```