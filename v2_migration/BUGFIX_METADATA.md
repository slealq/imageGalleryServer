# Bug Fix: SQLAlchemy Metadata Column Name Conflict

## Issue
SQLAlchemy's Declarative base reserves the `metadata` attribute for table metadata. Using `metadata` as a column name causes the error:

```
sqlalchemy.exc.InvalidRequestError: Attribute name 'metadata' is reserved when using the Declarative API.
```

## Solution
Renamed the `metadata` column to `extra_metadata` in all affected models:

### Files Changed

1. **Database Models:**
   - `src/models/database/photoset.py` - `metadata` → `extra_metadata`
   - `src/models/database/image.py` - `metadata` → `extra_metadata`
   - `src/models/database/embedding.py` - `metadata` → `extra_metadata`

2. **Pydantic Schemas:**
   - `src/models/schemas/photoset.py` - Updated `PhotosetBase` and `PhotosetUpdate`
   - `src/models/schemas/image.py` - Updated `ImageBase` and `ImageUpdate`

3. **Test Files:**
   - `scripts/test_foundation.py` - Updated test photoset creation

### Note
The `caption.py` model was NOT changed because it uses `generator_metadata`, which doesn't conflict with SQLAlchemy's reserved `metadata` attribute.

## Testing
After this fix, you can run:

```bash
alembic upgrade head
```

This will successfully create all database tables without errors.

