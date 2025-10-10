#!/usr/bin/env python3

import torch
from datasets import load_dataset, Audio
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

# --- NEW: PEFT / LoRA imports ---
from peft import LoraConfig, get_peft_model, TaskType


class ParakeetDataCollator:
    """Data collator for Parakeet CTC training - processes audio and text."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        """
        Each feature should have:
          - "audio": audio dict with array and sampling_rate
          - "text":  transcription string
        """
        texts = [f["text"] for f in features]
        audios = [f["audio"]["array"] for f in features]

        # Return tensors and pad in-batch
        batch = self.processor(
            audio=audios,
            text=texts,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Mask out padding in labels for CTC loss
        if "labels" in batch and hasattr(self.processor, "tokenizer"):
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                batch["labels"] = batch["labels"].masked_fill(batch["labels"] == pad_id, -100)

        return batch


def load_and_prepare_dataset():
    """Load and prepare dataset for training."""
    dataset_name = "hf-internal-testing/librispeech_asr_dummy"
    dataset_config = "clean"

    print(f"Loading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split="validation")

    # Cast audio to 16kHz (required for Parakeet)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Use same split for train and eval for quick testing
    train_dataset = dataset
    eval_dataset = dataset

    return train_dataset, eval_dataset


def main():
    # Configuration
    model_checkpoint = "nvidia/parakeet-ctc-1.1b"
    output_dir = "./parakeet-finetuned-lora"

    # Set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")

    # Load processor and model
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForCTC.from_pretrained(model_checkpoint)

    # --- NEW: Add LoRA ---
    # These target modules hit typical attention and FFN linear layers across many encoder models.
    # If you want to be more surgical, inspect model.named_modules() and narrow this list.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["linear"],
    )
    model = get_peft_model(model, lora_config)
    # Optional but handy:
    model.print_trainable_parameters()

    # (Optional) enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset()

    # Setup data collator
    data_collator = ParakeetDataCollator(processor)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        eval_steps=50 if eval_dataset else None,
        save_steps=50,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=1,
        ddp_find_unused_parameters=False,
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save LoRA adapters and processor
    print(f"Saving LoRA adapters to {output_dir}")
    trainer.save_model()              # saves PEFT adapters when model is a PEFT model
    processor.save_pretrained(output_dir)

    # Final evaluation
    if eval_dataset:
        results = trainer.evaluate()
        print(f"Final evaluation results: {results}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
