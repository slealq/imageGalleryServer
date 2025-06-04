from enum import Enum
from pathlib import Path

class CaptionGeneratorType(Enum):
    DUMMY = "dummy"
    UNSLOTH = "unsloth"

# Configuration settings
IMAGES_DIR = Path("/Users/stuartleal/gallery-project/images")
IMAGES_PER_PAGE = 10

# Caption generation settings
CAPTION_GENERATOR = CaptionGeneratorType.DUMMY  # Change to UNSLOTH to use AI generation

# Unsloth model settings (only used if CAPTION_GENERATOR is UNSLOTH)
UNSLOTH_MODEL_NAME = "unsloth/Llama-3.2-11B-Vision-Instruct"
UNSLOTH_MAX_SEQ_LENGTH = 2048
UNSLOTH_LOAD_IN_4BIT = True 