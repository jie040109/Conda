# Conda

Official PyTorch implementation of the paper:  
[“Conda: Column-Normalized Adam for Training Large Language Models Faster”](https://arxiv.org/abs/2509.24218)

---

## 📥 Installation

```bash
git clone https://github.com/jie040109/Conda.git
cd Conda
pip install -e .
```

## 🚀 Examples

This repository includes two example training setups using `conda_torch`:

- `examples/gpt2/` — GPT-2 pre-training on Openwebtext  
- `examples/llama/` — LLaMA pre-training on C4

Below are the exact steps to reproduce both examples.

---

## ✅ 1. LLaMA
### **Step 1 — Install dependencies**
```bash
cd examples/llama
conda create -n llama python=3.10
conda activate llama
pip install -r requirements.txt
```
### **Step 2 — Prepare C4 datasets**
```bash
bash download_c4.sh 
```

### **Step 3 — Conda for LLaMA pre-training**
```bash
# llama-60m
bash scripts/llama_60m_conda.sh
# llama-130m
bash scripts/llama_130m_conda.sh
# llama-350m
bash scripts/llama_350m_conda.sh
# llama-1b
bash scripts/llama_1b_conda.sh
```

### **Step 4 — Other optimizers for LLaMA pre-training**
Scripts for alternative optimizers (AdamW, Muon, SOAP, Adafactor) are located in:
```bash
examples/llama/scripts/
```
Run them in a similar manner, eg.
```bash
bash scripts/llama_60m_muon.sh
```

## ✅ 2. GPT-2 

### **Step 1 — Install dependencies**
```bash
cd examples/gpt2
conda env create -f environment.yml
conda activate gpt2
```

### **Step 2 — Prepare Openwebtext datasets**

```bash
python data/openwebtext/prepare.py
```

### **Step 3 — Conda for GPT-2 pre-training**
```bash
# gpt2-125m
bash scripts/train_gpt2_125m_conda.sh
# gpt2-355m
bash scripts/train_gpt2_355m_conda.sh
```

### **Step 4 — Other optimizers for GPT-2 pre-training**
Scripts for alternative optimizers (AdamW, Muon, SOAP, Adafactor) are located in:
```bash
examples/gpt2/scripts/
```
Run them in a similar manner, eg.
```bash
bash scripts/train_gpt2_125m_muon.sh
```