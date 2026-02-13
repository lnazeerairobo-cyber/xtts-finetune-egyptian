#!/usr/bin/env python3
"""
XTTS v2 Inference Script for Egyptian Arabic
Uses fine-tuned model for text-to-speech synthesis
"""

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import gradio as gr


def load_model(model_path: str):
    """Load fine-tuned XTTS v2 model"""
    print(f"Loading model from: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    return model, processor


def text_to_speech(text: str, model, processor, language: str = "egyptian"):
    """Generate speech from text"""
    if not text.strip():
        return None, "Please enter some text to convert to speech."
    
    try:
        # Prepare inputs
        inputs = processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        # Generate speech
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                language=language,
                do_sample=True,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode output
        speech = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return speech, None
    
    except Exception as e:
        return None, f"Error generating speech: {str(e)}"


def create_gradio_interface(model_path: str):
    """Create Gradio interface for text-to-speech"""
    model, processor = load_model(model_path)
    
    def generate_speech(text, language):
        speech, error = text_to_speech(text, model, processor, language)
        if error:
            return None, error
        return speech, None
    
    interface = gr.Interface(
        fn=generate_speech,
        inputs=[
            gr.Textbox(
                label="Text to Speak",
                placeholder="Enter Egyptian Arabic text here...",
                lines=3
            ),
            gr.Dropdown(
                choices=["egyptian", "arabic", "english"],
                value="egyptian",
                label="Language"
            )
        ],
        outputs=[
            gr.Audio(label="Generated Speech"),
            gr.Textbox(label="Status", interactive=False)
        ],
        title="XTTS v2 - Egyptian Arabic TTS",
        description="Generate speech in Egyptian Arabic using fine-tuned XTTS v2 model",
        examples=[
            ["السلام عليكم، كيف حالك؟", "egyptian"],
            ["أنا أحب البرمجة والتعلم الجديد", "egyptian"],
            ["Good morning, how are you?", "english"]
        ]
    )
    
    return interface


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="XTTS v2 TTS for Egyptian Arabic")
    parser.add_argument(
        "--model",
        type=str,
        default="./finetuned_xtts_egy",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public sharing link"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XTTS v2 Egyptian Arabic TTS")
    print("=" * 60)
    print(f"Loading model from: {args.model}")
    
    interface = create_gradio_interface(args.model)
    
    print("\nStarting Gradio interface...")
    print("Press Ctrl+C to stop")
    
    interface.launch(share=args.share, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
