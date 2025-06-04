from config import (
    CaptionGeneratorType,
    CAPTION_GENERATOR,
    UNSLOTH_LOAD_IN_4BIT
)

if CAPTION_GENERATOR == CaptionGeneratorType.UNSLOTH:
    from unsloth import FastVisionModel

from abc import ABC, abstractmethod
from PIL import Image
import torch
from pathlib import Path
from transformers import TextStreamer


class CaptionGenerator(ABC):
    @abstractmethod
    async def generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        pass

class DummyCaptionGenerator(CaptionGenerator):
    async def generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        """Generate a simple dummy caption."""
        if prompt:
            return f"A picture of {prompt}"
        return "A picture of something"

class UnslothCaptionGenerator(CaptionGenerator):
    def __init__(self):
        # Import Unsloth only when this class is instantiated
        
        # Define the model path (consider making this configurable in config.py)
        # Assuming the model is saved locally at this path after training/saving
        # model_path = Path("/mnt/c/playground/unsloth_model")
        
        #if not model_path.exists():
            #    raise FileNotFoundError(f"Unsloth vision model not found at {model_path}")

        #print(f"Loading Unsloth vision model from {model_path}...")
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct", # Load from local path
            load_in_4bit = UNSLOTH_LOAD_IN_4BIT, # Use config value
            use_gradient_checkpointing = "unsloth", # Recommended for long context, can be removed if not needed for inference
        )
        
        # Enable for inference
        FastVisionModel.for_inference(self.model)
        print("Unsloth vision model loaded and set for inference.")

    async def generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        """Generate a caption using the Unsloth vision model."""
        try:
            # Prepare the messages for the vision model, including the image
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt if prompt else "Describe the image."}
                ]}
                # Add an assistant turn if you want to guide the generation, e.g., starting phrase
                # {"role" : "assistant", "content" : [{"type" : "text",  "text"  : "The image shows"}]}, # Example
            ]

            # Apply chat template to get the input text string for the tokenizer
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize=False)
            
            # Tokenize the image and text inputs
            # Ensure tokenizer is called with both image and text
            inputs = self.tokenizer(
                image,
                input_text,
                add_special_tokens = False, # Handled by apply_chat_template
                return_tensors = "pt",
            ).to(self.model.device) # Move inputs to the same device as the model
            
            # Generate caption
            # Using parameters similar to the inference example in main.py
            text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
            generated_ids = self.model.generate(
                **inputs,
                streamer = text_streamer,
                max_new_tokens = 2048, # Consider making this configurable
                use_cache = True,
                temperature = 1.5, # Consider making this configurable
                min_p = 0.1,       # Consider making this configurable
                # do_sample = True, # Uncomment if using temperature/top_p
            )
            
            # Decode the generated text
            # Decode only the newly generated tokens if streamer is used and you want just the output
            # Otherwise, generated_ids will contain input + output tokens
            output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the caption - this might need adjustment based on actual model output format
            # The streamer typically handles skipping the prompt, but raw decode might not
            # A simple approach is to find the assistant's response start if a template is used
            # For simplicity, we'll rely on skip_prompt in streamer for now.

            return output_text.strip()
            
        except Exception as e:
            # Log the error and re-raise
            print(f"Error during Unsloth caption generation: {e}") # Consider using a proper logger
            raise Exception(f"Error generating caption with Unsloth: {str(e)}")

    async def stream_caption(self, image: Image.Image, prompt: str = None):
        """Stream the caption generation process."""
        try:
            # Prepare the messages for the vision model, including the image
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt if prompt else "Describe the image."}
                ]}
            ]

            # Apply chat template to get the input text string for the tokenizer
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize=False)
            
            # Tokenize the image and text inputs
            inputs = self.tokenizer(
                image,
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to(self.model.device)
            
            # Create a custom streamer that yields tokens
            class CaptionStreamer(TextStreamer):
                def __init__(self, tokenizer):
                    super().__init__(tokenizer)
                    self.current_text = ""
                    self.previous_text = ""  # Track previous text to detect changes
                    print("CaptionStreamer initialized")
                
                def put(self, value):
                    print(f"\n--- Token Debug ---")
                    print(f"Raw token value: {value}")
                    print(f"Token type: {type(value)}")
                    
                    # Handle tensor input
                    if isinstance(value, torch.Tensor):
                        # Convert tensor to list of integers
                        token_ids = value[0].tolist()  # Get first row and convert to list
                        print(f"Token IDs: {token_ids}")
                        
                        # Decode the tokens
                        token_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        print(f"Decoded token text: '{token_text}'")
                        
                        if token_text:
                            # Find the new text by comparing with previous
                            if token_text.startswith(self.previous_text):
                                new_text = token_text[len(self.previous_text):]
                                if new_text:  # Only yield if there's new text
                                    self.current_text = token_text
                                    print(f"Previous text: '{self.previous_text}'")
                                    print(f"New text: '{new_text}'")
                                    print(f"Current full text: '{self.current_text}'")
                                    # Yield the new text immediately
                                    yield new_text
                            else:
                                # If text doesn't start with previous, it might be a new generation
                                print("Text doesn't continue from previous, might be new generation")
                                self.current_text = token_text
                                yield token_text
                            
                            self.previous_text = token_text
                        else:
                            print("Empty token text, skipping")
                    else:
                        print(f"Unexpected token type: {type(value)}")
                        return ""
                    
                    return token_text

            print("\nCreating streamer and starting generation...")
            streamer = CaptionStreamer(self.tokenizer)
            
            # Generate with streaming
            print("Starting model generation...")
            generated_ids = self.model.generate(
                **inputs,
                streamer = streamer,
                max_new_tokens = 2048,
                use_cache = True,
                temperature = 1.5,
                min_p = 0.1,
            )
            print("Model generation completed")
            
            # The streamer will yield chunks as they are generated
            async for chunk in streamer:
                if chunk:  # Only yield non-empty chunks
                    yield chunk

        except Exception as e:
            print(f"Error during Unsloth caption streaming: {e}")
            raise Exception(f"Error streaming caption with Unsloth: {str(e)}")

def get_caption_generator() -> CaptionGenerator:
    """Factory function to get the appropriate caption generator based on configuration."""
    if CAPTION_GENERATOR == CaptionGeneratorType.UNSLOTH:
        #try:
        return UnslothCaptionGenerator()

            
    return DummyCaptionGenerator() 