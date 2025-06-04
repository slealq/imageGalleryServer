from abc import ABC, abstractmethod
from PIL import Image
from config import (
    CaptionGeneratorType,
    CAPTION_GENERATOR,
    UNSLOTH_MODEL_NAME,
    UNSLOTH_MAX_SEQ_LENGTH,
    UNSLOTH_LOAD_IN_4BIT
)

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
        import torch
        from unsloth import FastLanguageModel
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=UNSLOTH_MODEL_NAME,
            max_seq_length=UNSLOTH_MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=UNSLOTH_LOAD_IN_4BIT,
        )

    async def generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        """Generate a caption using the Unsloth model."""
        try:
            # Prepare the prompt
            if prompt:
                user_prompt = f"Please generate a caption for this image, focusing on: {prompt}"
            else:
                user_prompt = "Please generate a detailed caption for this image."
            
            # Generate caption using Unsloth
            inputs = self.tokenizer(
                user_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            # Decode the generated text
            caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the caption (remove the prompt and any extra whitespace)
            caption = caption.replace(user_prompt, "").strip()
            
            return caption
            
        except Exception as e:
            raise Exception(f"Error generating caption with Unsloth: {str(e)}")

def get_caption_generator() -> CaptionGenerator:
    """Factory function to get the appropriate caption generator based on configuration."""
    if CAPTION_GENERATOR == CaptionGeneratorType.UNSLOTH:
        return UnslothCaptionGenerator()
    return DummyCaptionGenerator() 