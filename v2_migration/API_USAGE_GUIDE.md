# API Usage Guide - Image Gallery v2

Quick reference for using the Image Gallery v2 REST API.

## Base URL

```
http://localhost:8002/api/v2
```

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## Authentication

Currently no authentication is required. This will be added in a future phase.

## Common Headers

```
Content-Type: application/json
```

## Response Headers

All responses include:

```
X-Request-ID: <uuid>       # Unique request identifier
X-Process-Time: <seconds>  # Processing time
```

---

## Photosets

### Create a Photoset

```http
POST /api/v2/photosets
Content-Type: application/json

{
  "name": "Summer Vacation 2024",
  "year": 2024,
  "source_url": "https://example.com/photos",
  "extra_metadata": {
    "location": "Hawaii",
    "photographer": "John Doe"
  }
}
```

**Response (201):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Summer Vacation 2024",
  "source_url": "https://example.com/photos",
  "date": null,
  "year": 2024,
  "original_archive_filename": null,
  "extra_metadata": {
    "location": "Hawaii",
    "photographer": "John Doe"
  },
  "created_at": "2024-10-14T12:00:00",
  "updated_at": "2024-10-14T12:00:00",
  "image_count": 0,
  "tags": []
}
```

### Get a Photoset

```http
GET /api/v2/photosets/{photoset_id}
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Summer Vacation 2024",
  "image_count": 45,
  "tags": ["vacation", "beach", "2024"],
  ...
}
```

---

## Images

### Get Image File

```http
GET /api/v2/images/{image_id}
```

**Response (200):**
- Content-Type: `image/jpeg`
- Body: Optimized JPEG image data
- Headers: `Cache-Control: public, max-age=31536000`

**Use in HTML:**
```html
<img src="http://localhost:8002/api/v2/images/{image_id}" alt="Image">
```

### Get Image Metadata

```http
GET /api/v2/images/{image_id}/metadata
```

**Response (200):**
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "photoset_id": "550e8400-e29b-41d4-a716-446655440000",
  "original_filename": "IMG_1234.jpg",
  "file_path": "550e8400-e29b-41d4-a716-446655440000/IMG_1234.jpg",
  "width": 3024,
  "height": 4032,
  "file_size": 2458624,
  "mime_type": "image/jpeg",
  "extra_metadata": {},
  "created_at": "2024-10-14T12:05:00",
  "updated_at": "2024-10-14T12:05:00",
  "has_caption": true,
  "has_crop": false,
  "tags": ["sunset", "beach"]
}
```

---

## Thumbnails

### Get Thumbnail

```http
GET /api/v2/images/{image_id}/thumbnail/{size}
```

**Sizes:**
- `small` - 256px
- `medium` - 512px
- `large` - 1024px

**Response (200):**
- Content-Type: `image/jpeg`
- Body: Thumbnail image data (cached)

**Auto-generation:**
Thumbnails are generated automatically on first request if they don't exist.

**Example:**
```html
<img src="http://localhost:8002/api/v2/images/{id}/thumbnail/medium" alt="Thumb">
```

---

## Captions

### Get Caption

```http
GET /api/v2/images/{image_id}/caption
```

**Response (200):**
```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "image_id": "660e8400-e29b-41d4-a716-446655440001",
  "caption": "A beautiful sunset over the ocean with palm trees in the foreground.",
  "generator_type": "unsloth",
  "generator_metadata": {
    "model": "llama-vision-11b",
    "temperature": 0.7
  },
  "created_at": "2024-10-14T12:10:00",
  "updated_at": "2024-10-14T12:10:00"
}
```

**Response (404):** If no caption exists

### Save/Update Caption

```http
POST /api/v2/images/{image_id}/caption
Content-Type: application/json

{
  "caption": "My custom caption for this image",
  "generator_type": "manual"
}
```

**Response (200):** Same as GET caption

### Generate Caption

```http
POST /api/v2/images/{image_id}/caption/generate
Content-Type: application/json

{
  "prompt": "Describe this image in detail"
}
```

**Response (200):**
```json
{
  "caption": "A beautiful sunset over the ocean with palm trees..."
}
```

**Note:** The caption is automatically saved to the database.

---

## Crops

### Get Crop Information

```http
GET /api/v2/images/{image_id}/crop
```

**Response (200):**
```json
{
  "crop_info": {
    "id": "880e8400-e29b-41d4-a716-446655440003",
    "image_id": "660e8400-e29b-41d4-a716-446655440001",
    "target_size": 512,
    "normalized_delta_x": 0.2,
    "normalized_delta_y": -0.1,
    "crop_file_path": "660e8400-e29b-41d4-a716-446655440001.png",
    "created_at": "2024-10-14T12:15:00",
    "updated_at": "2024-10-14T12:15:00"
  },
  "image_url": "/api/v2/images/660e8400-e29b-41d4-a716-446655440001/cropped"
}
```

**Response (404):** If no crop exists

### Get Cropped Image

```http
GET /api/v2/images/{image_id}/cropped
```

**Response (200):**
- Content-Type: `image/png`
- Body: Cropped image data

### Create/Update Crop

```http
POST /api/v2/images/{image_id}/crop
Content-Type: application/json

{
  "target_size": 512,
  "normalized_deltas": {
    "x": 0.2,
    "y": -0.1
  }
}
```

**Normalized Deltas:**
- Range: -1.0 to 1.0
- Represents the crop center offset
- x: -1 (left) to 1 (right)
- y: -1 (top) to 1 (bottom)

**Response (200):**
- Content-Type: `image/png`
- Body: Newly generated cropped image

---

## Tags

### List All Tags

```http
GET /api/v2/tags
```

**Response (200):**
```json
{
  "tags": [
    {
      "id": "990e8400-e29b-41d4-a716-446655440004",
      "name": "sunset",
      "tag_type": "custom",
      "created_at": "2024-10-14T12:20:00"
    },
    {
      "id": "aa0e8400-e29b-41d4-a716-446655440005",
      "name": "Emma Watson",
      "tag_type": "actor",
      "created_at": "2024-10-14T12:21:00"
    }
  ],
  "tags_by_type": {
    "custom": [...],
    "actor": [...],
    "photoset": [...],
    "image": [...]
  },
  "total": 42
}
```

### Add Tag to Image

```http
POST /api/v2/images/{image_id}/tags
Content-Type: application/json

{
  "tag_name": "sunset",
  "tag_type": "custom"
}
```

**Tag Types:**
- `custom` - User-defined tags
- `image` - Image-specific tags
- `photoset` - Photoset-level tags
- `actor` - Actor/person tags

**Response (200):**
```json
{
  "id": "990e8400-e29b-41d4-a716-446655440004",
  "name": "sunset",
  "tag_type": "custom",
  "created_at": "2024-10-14T12:20:00"
}
```

**Note:** Tags are automatically created if they don't exist.

---

## Health Check

### Check Service Health

```http
GET /api/v2/health
```

**Response (200):**
```json
{
  "status": "healthy",
  "database": "healthy",
  "cache": "healthy"
}
```

**Possible Values:**
- `healthy` - Service is operational
- `unhealthy` - Service is down
- `unavailable` - Service is not configured

---

## Error Responses

### 404 Not Found

```json
{
  "detail": "Image with ID 660e8400-e29b-41d4-a716-446655440001 not found"
}
```

### 400 Bad Request

```json
{
  "detail": "Failed to generate crop: Invalid image data"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "year"],
      "msg": "ensure this value is greater than or equal to 1900",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

## Code Examples

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8002/api/v2"

# Create photoset
response = requests.post(
    f"{BASE_URL}/photosets",
    json={
        "name": "My Photos",
        "year": 2024
    }
)
photoset = response.json()
print(f"Created photoset: {photoset['id']}")

# Get image
response = requests.get(f"{BASE_URL}/images/{image_id}")
with open("image.jpg", "wb") as f:
    f.write(response.content)

# Generate caption
response = requests.post(
    f"{BASE_URL}/images/{image_id}/caption/generate",
    json={"prompt": "Describe this image"}
)
caption = response.json()["caption"]
print(f"Generated caption: {caption}")
```

### JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:8002/api/v2";

// Create photoset
const createPhotoset = async () => {
  const response = await fetch(`${BASE_URL}/photosets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: "My Photos",
      year: 2024
    })
  });
  const photoset = await response.json();
  console.log("Created:", photoset.id);
};

// Get thumbnail
const getThumbnail = (imageId) => {
  return `${BASE_URL}/images/${imageId}/thumbnail/medium`;
};

// Add tag
const addTag = async (imageId, tagName) => {
  const response = await fetch(`${BASE_URL}/images/${imageId}/tags`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tag_name: tagName,
      tag_type: "custom"
    })
  });
  return response.json();
};
```

### cURL

```bash
# Health check
curl http://localhost:8002/api/v2/health

# Create photoset
curl -X POST http://localhost:8002/api/v2/photosets \
  -H "Content-Type: application/json" \
  -d '{"name": "Test", "year": 2024}'

# Get image
curl http://localhost:8002/api/v2/images/{id} -o image.jpg

# Generate caption
curl -X POST http://localhost:8002/api/v2/images/{id}/caption/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe"}' \
  | jq '.caption'
```

---

## Best Practices

1. **Use UUIDs**: All IDs are UUIDs, not integers
2. **Check metadata first**: Use `/images/{id}/metadata` before fetching the file
3. **Cache thumbnails**: Thumbnail URLs are stable and cacheable
4. **Handle 404s gracefully**: Not all images have captions or crops
5. **Use appropriate thumbnail sizes**: Small for lists, medium for previews, large for lightboxes
6. **Batch operations**: Use pagination for large photosets (TODO: implement)

---

## Rate Limiting

Currently no rate limiting is implemented. This will be added in a future phase.

---

## Versioning

API version is included in the URL: `/api/v2/...`

Future versions will be: `/api/v3/...`, etc.

---

**For complete API schema, visit**: http://localhost:8002/docs

