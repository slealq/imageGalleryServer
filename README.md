# Image Gallery Service

A FastAPI-based service for managing and processing images with optional AI-powered caption generation.

## Features

- Image management (upload, list, retrieve)
- Image cropping with customizable target sizes
- Optional AI-powered caption generation
- Image export functionality
- RESTful API endpoints

## Prerequisites

- Python 3.12+
- FastAPI
- Pillow
- PyTorch (optional, for AI caption generation)
- Unsloth (optional, for AI caption generation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gallery-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running
wsl
sudo mount -t drvfs D: /mnt/d
# pass 9571
cd
source unsloth/bin/activate
python /mnt/c/playground/imageGalleryServer/main.py



## Configuration

The service uses the following configuration:

- `IMAGES_DIR`: Directory where images are stored (default: `/Users/stuartleal/gallery-project/images`)
- `IMAGES_PER_PAGE`: Number of images per page in pagination (default: 10)
- `CAPTION_GENERATOR`: Type of caption generator to use (`DUMMY` or `UNSLOTH`)

## API Endpoints

### Image Management

- `GET /images`: List images with pagination
  - Query parameters:
    - `page`: Page number (default: 1)
    - `page_size`: Images per page (default: 10)
  - Returns: List of images with metadata

- `GET /images/{image_id}`: Get a specific image
  - Returns: Image file

### Caption Management

- `GET /images/{image_id}/caption`: Get image caption
  - Returns: Caption text

- `POST /images/{image_id}/caption`: Save image caption
  - Body: `{"caption": "string"}`

- `POST /images/{image_id}/generate-caption`: Generate caption using AI
  - Query parameters:
    - `prompt`: Optional prompt for caption generation
  - Returns: Generated caption

### Image Cropping

- `GET /images/{image_id}/preview/{target_size}`: Get image preview
  - Returns: Scaled image preview

- `POST /images/{image_id}/crop`: Crop image
  - Body:
    ```json
    {
      "targetSize": number,
      "normalizedDeltas": {
        "x": number,
        "y": number
      }
    }
    ```
  - Returns: Cropped image

### Export

- `POST /api/export-images`: Export selected images
  - Body:
    ```json
    {
      "imageIds": ["string"]
    }
    ```
  - Returns: ZIP file containing selected images and their captions

## AI Caption Generation

The service supports two modes for caption generation:

1. **Dummy Mode**: Generates simple, predefined captions
   - No additional dependencies required
   - Fast and lightweight
   - Good for testing and development
   - Example: "A picture of something" or "A picture of {prompt}"

2. **AI Mode**: Uses Unsloth's Llama 3.2 Vision model
   - Requires NVIDIA or Intel GPU
   - More sophisticated captions
   - Higher resource requirements
   - Dependencies:
     - PyTorch
     - Unsloth
     - Transformers
     - Accelerate

To switch between modes, modify the `CAPTION_GENERATOR` setting in `config.py`:
```python
# For dummy mode (default)
CAPTION_GENERATOR = CaptionGeneratorType.DUMMY

# For AI mode
CAPTION_GENERATOR = CaptionGeneratorType.UNSLOTH
```

## File Structure

```
gallery-project/
├── image_server/
│   ├── main.py              # Main FastAPI application
│   ├── caption_generator.py # Caption generation logic
│   ├── config.py           # Configuration settings
│   └── requirements.txt    # Python dependencies
├── images/                 # Image storage directory
└── README.md              # This documentation
```

## Running the Service

1. Start the server:
```bash
cd image_server
python main.py
```

2. The server will start at `http://localhost:4322`

## Development

### Adding New Features

1. Create new endpoints in `main.py`
2. Add corresponding models in the Models section
3. Implement business logic in separate modules
4. Update documentation

### Testing

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Support

For support, please [open an issue](repository-issues-url) or contact [your contact information]. 