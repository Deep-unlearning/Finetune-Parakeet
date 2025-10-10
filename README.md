# Finetune Voxtral for ASR with Transformers 🤗

This repository fine-tunes the [Parakeet](https://github.com/NVIDIA/Parakeet) speech model on conversational speech datasets using the Hugging Face `transformers` and `datasets` libraries.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Deep-unlearning/Finetune-Parakeet.git
cd Finetune-Voxtral-ASR
```

### Step 2: Set up environment

Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
```

</details>

<details>
<summary>🐍 Using pip</summary>

```bash
python -m venv .venv --python 3.10 && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

</details>

## Dataset Preparation

If you want to swap to a different dataset, ensure after loading you still have:

* an **`audio`** column (cast to `Audio(sampling_rate=16000)`), and
* a **`text`** column (the reference transcription).

If your dataset uses different column names, map them to `audio` and `text` before returning.

## Training

Run the training script:

```bash
uv run train.py
```

Logs and checkpoints will be saved under the `outputs/` directory by default.

## Training with LoRA

You can also run the training script with LoRA:

```bash
uv run train_lora.py
```

**Happy fine-tuning Parakeet!** 🚀