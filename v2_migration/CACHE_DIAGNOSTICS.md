# Cache Performance Diagnostics

## Overview

All image endpoints now include detailed performance diagnostics in response headers to help diagnose slow loads and verify Redis cache usage.

## Diagnostic Headers

When you request an image from `/images/{image_id}`, you'll receive these headers:

### Cache Status

- **`X-Cache-Status`**: `HIT` or `MISS`
  - `HIT` = Image served from Redis cache (fast!)
  - `MISS` = Image loaded from disk (slower, then cached for next time)

- **`X-Cache-Enabled`**: `True` or `False`
  - Whether caching is enabled for this request

### Timing Breakdown (all in milliseconds)

- **`X-Timing-ID-Lookup`**: Time to convert string ID to UUID
  - Only relevant for legacy endpoints with string IDs
  - Should be < 10ms typically

- **`X-Timing-Cache-Check`**: Time to check Redis cache
  - Should be < 5ms for local Redis
  - If > 100ms, Redis might be slow or network issue

- **`X-Timing-DB-Lookup`**: Time to query PostgreSQL for image metadata
  - Only happens on cache miss
  - Should be < 50ms typically
  - If > 500ms, database might be slow

- **`X-Timing-File-Read`**: Time to read image file from disk
  - Only happens on cache miss
  - Can be slow for large images or slow storage
  - **This is often the bottleneck!**

- **`X-Timing-Image-Processing`**: Time to convert/optimize image
  - Only happens on cache miss
  - Includes format conversion (RGBA → RGB) and JPEG encoding
  - Can be slow for very large images

- **`X-Timing-Cache-Write`**: Time to write image to Redis cache
  - Only happens on cache miss
  - Should be < 10ms for local Redis
  - If > 100ms, Redis write might be slow

- **`X-Timing-Backend-Total`**: Total backend processing time
  - Sum of all operations
  - **Cache HIT**: Should be < 10ms
  - **Cache MISS**: Can be several seconds for large images

- **`X-Timing-Request-Total`**: Total request time (including all overhead)
  - Includes routing, ID lookup, and all processing

## Example Headers

### Fast Response (Cache Hit)
```
X-Cache-Status: HIT
X-Cache-Enabled: True
X-Timing-ID-Lookup: 2.34
X-Timing-Cache-Check: 3.21
X-Timing-DB-Lookup: 0
X-Timing-File-Read: 0
X-Timing-Image-Processing: 0
X-Timing-Cache-Write: 0
X-Timing-Backend-Total: 3.21
X-Timing-Request-Total: 5.67
```
**Analysis**: Perfect! Served from cache in 3ms, total request only 5.67ms.

### Slow Response (Cache Miss, Large Image)
```
X-Cache-Status: MISS
X-Cache-Enabled: True
X-Timing-ID-Lookup: 3.45
X-Timing-Cache-Check: 2.11
X-Timing-DB-Lookup: 45.23
X-Timing-File-Read: 18234.56
X-Timing-Image-Processing: 2345.67
X-Timing-Cache-Write: 123.45
X-Timing-Backend-Total: 20751.02
X-Timing-Request-Total: 20755.89
```
**Analysis**: Slow! 20+ seconds total. Breakdown:
- **File Read**: 18.2 seconds (BOTTLENECK!)
- **Image Processing**: 2.3 seconds
- **Other operations**: < 200ms

**Issue**: File read is extremely slow (18+ seconds). Possible causes:
1. Image file is on slow storage (network drive, slow disk)
2. Image file is very large
3. Disk I/O is saturated
4. Storage path configuration issue

## Checking Your Responses

### Using Browser DevTools
1. Open DevTools (F12)
2. Go to Network tab
3. Load an image
4. Click on the image request
5. Look at Response Headers

### Using curl
```bash
curl -I http://192.168.68.71:8002/images/330f19f0-c0d2-4aa3-8a38-d4059ad70c65

# Look for X-Cache-Status and X-Timing-* headers
```

### Using Python
```python
import requests

response = requests.get('http://192.168.68.71:8002/images/330f19f0-c0d2-4aa3-8a38-d4059ad70c65')

print(f"Cache: {response.headers.get('X-Cache-Status')}")
print(f"Backend Time: {response.headers.get('X-Timing-Backend-Total')}ms")
print(f"File Read: {response.headers.get('X-Timing-File-Read')}ms")
```

## Common Issues and Solutions

### Issue: 20-30 second load times

**Check the headers first!**

#### If `X-Timing-File-Read` is very high (> 5000ms):
- **Problem**: Slow disk I/O
- **Solutions**:
  1. Check if `storage_root` in config points to a fast local disk
  2. Verify images aren't on a slow network drive
  3. Check disk I/O with `iostat` or Task Manager
  4. Consider moving images to SSD if on HDD
  5. Check file permissions (Windows file locking issues?)

#### If `X-Timing-Image-Processing` is very high (> 3000ms):
- **Problem**: Large image or slow CPU
- **Solutions**:
  1. Check image dimensions (might be 10000x10000px)
  2. Pre-process images to reasonable size (< 4000px)
  3. Increase CPU resources
  4. Consider disabling optimization: change `optimize=True` to `optimize=False`

#### If `X-Timing-Cache-Check` is very high (> 100ms):
- **Problem**: Redis connection issues
- **Solutions**:
  1. Check Redis is running: `redis-cli ping`
  2. Check Redis connection in config
  3. Check network latency to Redis server
  4. Check Redis memory usage: `redis-cli info memory`

#### If `X-Cache-Status` is always MISS:
- **Problem**: Cache not working or being bypassed
- **Solutions**:
  1. Check Redis is running
  2. Check Redis connection URL in config
  3. Check Redis logs for errors
  4. Verify cache key TTL isn't too short
  5. Check Redis memory isn't full (`maxmemory` policy)

### Issue: First load slow, subsequent loads fast
- **This is expected!**
- First load: Cache MISS → reads from disk → caches in Redis
- Subsequent loads: Cache HIT → served from Redis in < 10ms
- **Solution**: Pre-warm cache by loading images once

### Issue: All cache hits but still slow (network latency)
- **Problem**: Network latency between client and server
- **Check**: `X-Timing-Backend-Total` vs actual client-side load time
- If backend is fast (< 100ms) but client sees seconds, it's network
- **Solutions**:
  1. Check network connection quality
  2. Consider image compression
  3. Use CDN for production
  4. Check firewall/antivirus isn't scanning traffic

## Redis Cache Verification

### Check if Redis is being used:
```bash
# Connect to Redis
redis-cli

# Monitor cache operations in real-time
MONITOR

# In another terminal, load an image
# You should see GET and SET operations for keys like "image:330f19f0-..."
```

### Check cache size:
```bash
redis-cli

# Count cached images
KEYS image:*

# Get memory info
INFO memory
```

### Clear cache (if needed):
```bash
redis-cli

# Clear all image cache
KEYS image:* | xargs redis-cli DEL

# Or flush entire Redis (careful!)
FLUSHALL
```

## Performance Targets

### Optimal Performance
- Cache HIT: < 10ms backend time
- Cache MISS (small image < 1MB): < 500ms
- Cache MISS (large image < 10MB): < 2000ms

### Concerning Performance
- Cache HIT: > 100ms (Redis issue)
- Cache MISS: > 5000ms (Storage/processing issue)
- Any single operation > 5000ms needs investigation

## Next Steps

1. **Load the slow image** and check the diagnostic headers
2. **Identify the bottleneck** (File Read? Processing? Cache?)
3. **Take action** based on the bottleneck identified
4. **Verify** by loading the image again (should be cached)
5. **Report findings** with the diagnostic headers for further help

## Questions to Ask

When reporting slow loads, include:
- What is `X-Cache-Status`? (HIT or MISS?)
- What is `X-Timing-File-Read`? (Is file read the bottleneck?)
- What is `X-Timing-Backend-Total`? (How fast is the backend?)
- Is it slow on first load only, or every time?
- What is the image file size and dimensions?
- Where is `storage_root` pointing to? (Local SSD? Network drive?)

