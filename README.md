# Finetune-Parakeet

Fine-tuning scripts for NVIDIA Parakeet CTC ASR models using Hugging Face Transformers.

## Overview

This repository provides tools for fine-tuning the NVIDIA Parakeet CTC model for automatic speech recognition tasks. It supports both full fine-tuning and parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).

## Model

- **Base Model**: `nvidia/parakeet-ctc-1.1b`
- **Architecture**: Fast Conformer encoder with CTC decoder
- **Vocabulary Size**: 1025 tokens
- **Parameters**: 1.1B parameters

## Installation

```bash
git clone https://github.com/your-username/Finetune-Parakeet.git
cd Finetune-Parakeet
pip install -r requirements.txt
```

## Dataset Format

Your dataset should have two key fields:
- `audio`: Audio data (float32 numpy array at 16kHz or dict with 'array' and 'sampling_rate')
- `text`: Reference transcription (string)

Example dataset structure:
```python
{
    "audio": {
        "array": [...],  # float32 numpy array
        "sampling_rate": 16000
    },
    "text": "hello world"
}
```

## Training Scripts

### Full Fine-tuning

Use `train.py` for full model fine-tuning:

```bash
python train.py \
    --dataset_name "mozilla-foundation/common_voice_11_0" \
    --dataset_config "en" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --fp16 \
    --output_dir "./parakeet-finetuned"
```

### LoRA Fine-tuning

Use `train_lora.py` for parameter-efficient fine-tuning:

```bash
python train_lora.py \
    --dataset_name "mozilla-foundation/common_voice_11_0" \
    --dataset_config "en" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir "./parakeet-lora"
```

## Training Arguments

### Dataset Arguments
- `--dataset_name`: Name of the dataset on Hugging Face Hub
- `--dataset_config`: Dataset configuration/subset name
- `--audio_column`: Name of the audio column (default: "audio")
- `--text_column`: Name of the text column (default: "text")
- `--train_split`: Name of the training split (default: "train")
- `--eval_split`: Name of the evaluation split (default: "validation")

### Training Arguments
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Training batch size per device (default: 8)
- `--per_device_eval_batch_size`: Evaluation batch size per device (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--warmup_steps`: Number of warmup steps (default: 500)
- `--fp16`: Use mixed precision training
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)

### LoRA-specific Arguments (train_lora.py)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.1)
- `--target_modules`: Target modules for LoRA (default: all linear layers)

## Quick Start

### Simple Training (Recommended for Testing)
Use the lean training script with dummy dataset:
```bash
python train_simple.py
```

This will train on the `hf-internal-testing/librispeech_asr_dummy` dataset for quick testing.

## Example Usage

### Quick Test with Dummy Dataset
```bash
# Full training
python train.py \
    --dataset_name "hf-internal-testing/librispeech_asr_dummy" \
    --dataset_config "clean" \
    --train_split "validation" \
    --eval_split "validation" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --output_dir "./parakeet-dummy-test"

# LoRA training
python train_lora.py \
    --dataset_name "hf-internal-testing/librispeech_asr_dummy" \
    --dataset_config "clean" \
    --train_split "validation" \
    --eval_split "validation" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --lora_r 8 \
    --output_dir "./parakeet-lora-dummy-test"
```

### Common Voice English
```bash
python train.py \
    --dataset_name "mozilla-foundation/common_voice_11_0" \
    --dataset_config "en" \
    --audio_column "audio" \
    --text_column "sentence" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --fp16 \
    --output_dir "./parakeet-cv-en"
```

## Features

- **CTC Loss**: Optimized for Connectionist Temporal Classification
- **Audio Preprocessing**: Automatic resampling to 16kHz
- **WER Evaluation**: Word Error Rate computation during training
- **Mixed Precision**: FP16 training support for faster training
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Early Stopping**: Prevent overfitting with automatic early stopping
- **Checkpoint Recovery**: Resume training from checkpoints
- **LoRA Support**: Parameter-efficient fine-tuning option

## Model Output

After training, the fine-tuned model can be used for inference:

```python
from transformers import AutoModelForCTC, AutoProcessor
import torch

# Load fine-tuned model
processor = AutoProcessor.from_pretrained("./parakeet-finetuned")
model = AutoModelForCTC.from_pretrained("./parakeet-finetuned")

# Process audio
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")

# Generate transcription
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

print(transcription)
```

## Performance Tips

1. **Batch Size**: Start with smaller batch sizes (4-8) and increase gradually
2. **Learning Rate**: Use lower learning rates (1e-5 to 1e-4) for stability
3. **Mixed Precision**: Enable `--fp16` for faster training on modern GPUs
4. **Gradient Accumulation**: Use to simulate larger batch sizes on limited memory
5. **LoRA**: Use for faster training with less memory usage

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU (recommended)

## License

This project is licensed under the MIT License.