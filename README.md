# XTTS v2 Egyptian Arabic Fine-Tuning

Fine-tuning script for Coqui XTTS v2 model using Egyptian Arabic speech dataset.

## Overview

This project fine-tunes the XTTS v2 text-to-speech model on the Egy-Speech-DeepClean dataset to improve Egyptian Arabic pronunciation and naturalness.

## Dataset

- **Dataset**: [MohamedGomaa30/Egy-Speech-DeepClean_v0](https://huggingface.co/datasets/MohamedGomaa30/Egy-Speech-DeepClean_v0)
- **Language**: Egyptian Arabic
- **Purpose**: Improve TTS quality for Egyptian dialect

## Features

- ✅ Fine-tunes XTTS v2 with LoRA for efficient training
- ✅ Supports 16-bit precision training
- ✅ Configurable batch size and learning rate
- ✅ Automatic gradient accumulation for large models
- ✅ Saves fine-tuned model with configuration

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python finetune_xtts.py
```

### Custom Configuration

Edit the `config` dictionary in `finetune_xtts.py`:

```python
config = {
    "model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
    "dataset_path": "MohamedGomaa30/Egy-Speech-DeepClean_v0",
    "output_dir": "./finetuned_xtts_egy",
    "batch_size": 4,
    "learning_rate": 1e-4,
    "num_epochs": 3,
    # ... more options
}
```

## Model Output

After training, the fine-tuned model will be saved in the `output_dir`:

```
finetuned_xtts_egy/
├── config.json
├── model.safetensors
├── finetune_xtts.py
├── requirements.txt
└── training_config.json
```

## Fine-Tuning Parameters

- **LoRA Rank**: 16 (balance between performance and memory)
- **Learning Rate**: 1e-4 (default, can be adjusted)
- **Batch Size**: 4 (works with 16GB VRAM)
- **Gradient Accumulation**: 4 (simulates larger batch size)
- **Epochs**: 3 (adjust based on dataset size)

## Memory Requirements

- **Minimum**: 16GB VRAM
- **Recommended**: 24GB+ VRAM
- **CPU-only**: ~32GB RAM (slow training)

## License

This project follows the license of XTTS v2 and the dataset. Please review licenses before use.

## References

- [XTTS v2 Model](https://huggingface.co/tts_models/multilingual/multi-dataset/xtts_v2)
- [Egy-Speech-DeepClean Dataset](https://huggingface.co/datasets/MohamedGomaa30/Egy-Speech-DeepClean_v0)
- [Coqui XTTS](https://github.com/coqui-ai/TTS)

## Contributing

Contributions welcome! Please open an issue or pull request.
