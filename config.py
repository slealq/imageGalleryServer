from enum import Enum
from pathlib import Path

class CaptionGeneratorType(Enum):
    DUMMY = "dummy"
    UNSLOTH = "unsloth"

# Configuration settings
# IMAGES_DIR = Path("/Users/stuartleal/gallery-project/images")
IMAGES_DIR = Path("/mnt/d/TEST/images")
CROPS_DIR = Path("/mnt/d/TEST/crops")
CAPTIONS_DIRECTORY = Path("/mnt/d/TEST/captions")
PHOTOSET_METADATA_DIRECTORY = Path("/mnt/d/TEST/photoset_metadata")
IMAGE_METADATA_DIRECTORY = Path("/mnt/d/TEST/images_metadata")
METADATA_DIRECTORY = Path("/mnt/d/TEST/metadata")
IMAGES_PER_PAGE = 10

# Server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8001

# Feature Flags
PROFILING_ENABLED = True # Set to True to enable profiling on the /images endpoint
PROFILING_DIR = IMAGES_DIR / "profiling_results" # Directory to save profiling results

# Caption generation settings
CAPTION_GENERATOR = CaptionGeneratorType.UNSLOTH  # Change to UNSLOTH to use AI generation

# Unsloth model settings (only used if CAPTION_GENERATOR is UNSLOTH)
# UNSLOTH_MODEL_NAME = "unsloth/Llama-3.2-11B-Vision-Instruct" # Not used when loading from local path
# UNSLOTH_MAX_SEQ_LENGTH = 2048 # Not directly used in inference generate
UNSLOTH_LOAD_IN_4BIT = True 