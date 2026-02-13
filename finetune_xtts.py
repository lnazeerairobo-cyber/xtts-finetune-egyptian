#!/usr/bin/env python3
"""
XTTS v2 Fine-Tuning Script for Egyptian Arabic Speech
Dataset: MohamedGomaa30/Egy-Speech-DeepClean_v0
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes.optim import AdamW8bit
from accelerate import Accelerator
from tqdm import tqdm
import torch.nn as nn


class XTTSDataset(Dataset):
    """Custom dataset for XTTS v2 fine-tuning"""
    
    def __init__(self, dataset, processor, max_duration=10.0):
        self.dataset = dataset
        self.processor = processor
        self.max_duration = max_duration
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Load audio
        audio_path = sample['audio']['path']
        audio = self.processor.feature_extractor(
            audio_path,
            sampling_rate=24000,
            return_tensors="pt"
        ).input_features
        
        # Process text
        text = sample['text']
        if isinstance(text, str):
            text = text.lower()
        
        # Get language
        language = sample.get('language', 'egyptian')
        
        # Return processed data
        return {
            'input_features': audio.squeeze(0),
            'text': text,
            'language': language
        }


def load_model_and_processor(model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
    """Load XTTS v2 model and processor"""
    print(f"Loading model: {model_id}")
    
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor


def prepare_dataset(dataset_path: str, split: str = "train"):
    """Load and prepare dataset"""
    print(f"Loading dataset from: {dataset_path}")
    
    dataset = load_dataset(dataset_path, split=split)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    return dataset


def apply_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA fine-tuning"""
    print(f"Applying LoRA with rank={rank}, alpha={alpha}, dropout={dropout}")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def create_dataloader(dataset, processor, batch_size: int = 8):
    """Create dataloader for training"""
    print(f"Creating dataloader with batch_size={batch_size}")
    
    xtts_dataset = XTTSDataset(dataset, processor)
    
    def collate_fn(batch):
        input_features = torch.stack([item['input_features'] for item in batch])
        texts = [item['text'] for item in batch]
        languages = [item['language'] for item in batch]
        
        # Tokenize texts
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        
        return {
            'input_features': input_features,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['input_ids'],
            'language': languages
        }
    
    dataloader = DataLoader(
        xtts_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def train_step(model, batch, accelerator, optimizer, criterion, device):
    """Single training step"""
    model.train()
    
    input_features = batch['input_features'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    outputs = model(
        input_features=input_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    
    # Backward pass
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()


def main():
    # Configuration
    config = {
        "model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
        "dataset_path": "MohamedGomaa30/Egy-Speech-DeepClean_v0",
        "output_dir": "./finetuned_xtts_egy",
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 4
    }
    
    # Setup accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    
    print("=" * 60)
    print("XTTS v2 Egyptian Arabic Fine-Tuning")
    print("=" * 60)
    
    # Load model and processor
    model, processor = load_model_and_processor(config["model_id"])
    
    # Prepare dataset
    dataset = prepare_dataset(config["dataset_path"], split="train")
    
    # Apply LoRA
    model = apply_lora(
        model,
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"]
    )
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        processor,
        batch_size=config["batch_size"]
    )
    
    # Prepare optimizer
    optimizer = AdamW8bit(model.parameters(), lr=config["learning_rate"])
    scheduler = None  # Can add learning rate scheduler here
    
    # Prepare model with accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    # Training loop
    total_steps = len(dataloader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    global_step = 0
    
    print(f"\nStarting training for {config['num_epochs']} epochs")
    print(f"Total steps: {total_steps}")
    print(f"Device: {device}")
    print("=" * 60)
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            loss = train_step(
                model, batch, accelerator, optimizer, None, device
            )
            
            epoch_loss += loss
            global_step += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step": global_step
            })
            
            # Log training progress
            if global_step % 100 == 0:
                print(f"Step {global_step}: Loss = {loss:.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Save fine-tuned model
    print(f"\nSaving fine-tuned model to: {config['output_dir']}")
    
    model.save_pretrained(config["output_dir"])
    processor.save_pretrained(config["output_dir"])
    
    # Save training config
    config_path = os.path.join(config["output_dir"], "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Fine-tuned model saved successfully!")
    print(f"📁 Output directory: {config['output_dir']}")
    print(f"📄 Config saved to: {config_path}")


if __name__ == "__main__":
    main()
