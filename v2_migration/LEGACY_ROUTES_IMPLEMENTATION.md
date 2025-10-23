# Legacy Routes Implementation

This document describes the backward compatibility layer implemented in the v2 migration service to support all original API routes.

## Overview

The v2 service has been updated to support **all routes from the original service** at their original paths, ensuring zero-downtime migration for existing clients.

## Implementation Strategy

### 1. Dual Route Structure

- **New v2 routes**: Available under `/api/v2/*` prefix (UUID-based, modern architecture)
- **Legacy routes**: Available at original paths (string ID-based, backward compatible)

### 2. File Structure

- **`src/api/routes/legacy.py`**: Contains all legacy route implementations
- Routes are automatically included in `main.py` via the legacy router

## Implemented Routes

### Root Level Routes

✅ **GET /filters** - Get available filters (actors, tags, years)

### Image Routes

✅ **GET /images** - List images with pagination and filters
  - Parameters: `page`, `page_size`, `actor`, `tag`, `year`, `has_caption`, `has_crop`
  - Backward compatible with old response format

✅ **GET /images/batch** - Fetch multiple pages at once
  - Parameters: `start_page`, `num_pages`, `page_size`, plus all filter params
  
✅ **GET /images/{image_id}** - Get image file by string ID
  - Supports both UUID strings and filename-based lookups

### Caption Routes (prefix: /images)

✅ **POST /images/{image_id}/caption** - Save caption for an image
✅ **GET /images/{image_id}/caption** - Get caption for an image  
✅ **POST /images/{image_id}/generate-caption** - Generate caption using AI
✅ **POST /images/{image_id}/stream-caption** - Stream caption generation (SSE)

### Tag Routes (prefix: /images)

✅ **GET /images/{image_id}/tags** - Get all tags for an image
✅ **GET /images/{image_id}/custom-tags** - Get only custom tags
✅ **POST /images/{image_id}/tags** - Add tag to an image

### Crop Routes (prefix: /images)

✅ **GET /images/{image_id}/crop** - Get crop information
✅ **GET /images/{image_id}/cropped** - Get cropped image file
✅ **POST /images/{image_id}/crop** - Create/update crop
✅ **GET /images/{image_id}/preview/{target_size}** - Get scaled preview

### Utility Routes

✅ **POST /api/export-images** - Export images with crops and captions as ZIP
✅ **POST /cache/warmup** - Cache warming (stub for v2 compatibility)

## String ID to UUID Conversion

### Challenge

The old service used string-based image IDs (typically filenames without extensions), while v2 uses UUIDs. The legacy routes handle this conversion automatically.

### Solution

Implemented a smart lookup system in `ImageService.find_image_by_string_id()`:

1. **Try UUID parsing**: If the string is a valid UUID, use it directly
2. **Filename pattern search**: Search `original_filename` and `file_path` for matches
3. **First match wins**: Returns the first matching image

### Usage Example

```python
# Client sends request with string ID
GET /images/my-image-file-name

# Legacy router automatically:
1. Calls get_image_uuid_from_string_id("my-image-file-name", image_service)
2. Searches database for images matching the pattern
3. Converts to UUID
4. Calls v2 service with UUID
5. Returns response in old format
```

## Key Features

### 1. Request/Response Transformation

Legacy routes automatically transform between old and new formats:

```python
# Old format (what client sends)
{
  "imageIds": ["image1", "image2"],
  "caption": "A beautiful sunset"
}

# New format (what v2 service uses internally)
{
  "image_id": UUID("..."),
  "caption_text": "A beautiful sunset",
  "generator_type": "manual"
}
```

### 2. Error Handling

- 404 errors for images not found
- 400 errors for invalid requests
- Graceful fallback when features not available

### 3. Streaming Support

The legacy caption streaming endpoint properly implements Server-Sent Events (SSE) to match the old service behavior:

```python
@router.post("/images/{image_id}/stream-caption")
async def stream_caption_legacy(...):
    async def generate():
        async for chunk in caption_service.stream_caption(uuid_id, prompt):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Migration Notes

### For Frontend Developers

1. **No changes required**: All existing API calls will continue to work
2. **Optional migration path**: Gradually migrate to `/api/v2/*` endpoints with UUIDs
3. **Better performance**: New endpoints use database-backed caching and optimization

### For Backend Developers

1. **String ID lookup**: May be slower than direct UUID lookup
2. **Monitoring recommended**: Track usage of legacy vs new endpoints
3. **Deprecation timeline**: Plan to deprecate legacy routes after full migration

## Testing

To test legacy endpoints:

```bash
# Test filters
curl http://localhost:8000/filters

# Test image listing
curl http://localhost:8000/images?page=1&page_size=10

# Test image retrieval with string ID
curl http://localhost:8000/images/my-image-name

# Test caption
curl -X POST http://localhost:8000/images/my-image-name/caption \
  -H "Content-Type: application/json" \
  -d '{"caption": "Test caption"}'

# Test export
curl -X POST http://localhost:8000/api/export-images \
  -H "Content-Type: application/json" \
  -d '{"imageIds": ["image1", "image2"]}'
```

## Performance Considerations

### String ID Lookup Cost

- **UUID string**: Fast (direct database query)
- **Filename pattern**: Slower (requires LIKE query)
- **Recommendation**: Cache string_id → UUID mappings if performance is critical

### Optimization Options

1. **Add caching layer**: Cache string ID to UUID mappings in Redis
2. **Create ID mapping table**: Store legacy_id → uuid mappings in database
3. **Index optimization**: Ensure `original_filename` and `file_path` are indexed

## Route Comparison

| Feature | Old Service | V2 Service (legacy) | V2 Service (new) |
|---------|-------------|---------------------|------------------|
| Base Path | `/` | `/` (legacy) | `/api/v2/` |
| Image ID | String | String → UUID | UUID |
| Auth | None | None | Token-based (future) |
| Response Format | Custom | Same as old | OpenAPI standard |
| Pagination | page/page_size | page/page_size | skip/limit |

## Future Improvements

1. **Add deprecation headers**: Include `X-Deprecated: true` in legacy responses
2. **Usage analytics**: Track which clients still use legacy endpoints
3. **Gradual migration**: Provide client-side SDK for easy migration
4. **ID mapping service**: Dedicated service for string_id ↔ UUID conversion

## Conclusion

All original service routes are now available in v2 with full backward compatibility. This allows for seamless migration while maintaining the modern v2 architecture underneath.

