from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
import time
import shutil
import os
from datetime import datetime

from ..models.models import ExportRequest
from ..services.image_service import get_image_path
from config import IMAGES_DIR, EXPORT_DIR

router = APIRouter()

@router.post("/export")
async def export_images(request: Request, export_request: ExportRequest):
    """Export selected images to a new directory."""
    start_time = time.time()
    try:
        # Create export directory with timestamp
        dir_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = Path(EXPORT_DIR) / f"export_{timestamp}"
        export_path.mkdir(parents=True, exist_ok=True)
        dir_time = time.time() - dir_start
        
        # Copy images
        copy_start = time.time()
        copied_files = []
        for image_id in export_request.imageIds:
            image_path = get_image_path(image_id, IMAGES_DIR)
            if image_path and image_path.exists():
                dest_path = export_path / image_path.name
                shutil.copy2(image_path, dest_path)
                copied_files.append(image_path.name)
        copy_time = time.time() - copy_start
        
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Export operation took {total_time:.3f}s (dir creation: {dir_time:.3f}s, file copy: {copy_time:.3f}s)")
        
        return {
            "message": f"Successfully exported {len(copied_files)} images",
            "exportPath": str(export_path),
            "exportedFiles": copied_files
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Error in export operation after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/exports")
async def list_exports(request: Request):
    """List all export directories."""
    start_time = time.time()
    try:
        # List export directories
        list_start = time.time()
        exports = []
        for item in Path(EXPORT_DIR).iterdir():
            if item.is_dir() and item.name.startswith("export_"):
                stats = item.stat()
                exports.append({
                    "name": item.name,
                    "path": str(item),
                    "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    "size": sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                })
        list_time = time.time() - list_start
        
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Export listing took {total_time:.3f}s (directory scan: {list_time:.3f}s)")
        
        return {"exports": exports}
        
    except Exception as e:
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Error in export listing after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/exports/{export_name}")
async def delete_export(request: Request, export_name: str):
    """Delete an export directory."""
    start_time = time.time()
    try:
        # Validate export directory
        validate_start = time.time()
        export_path = Path(EXPORT_DIR) / export_name
        if not export_path.exists() or not export_path.is_dir() or not export_name.startswith("export_"):
            raise HTTPException(status_code=404, detail="Export directory not found")
        validate_time = time.time() - validate_start
        
        # Delete directory
        delete_start = time.time()
        shutil.rmtree(export_path)
        delete_time = time.time() - delete_start
        
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Export deletion took {total_time:.3f}s (validation: {validate_time:.3f}s, deletion: {delete_time:.3f}s)")
        
        return {"message": f"Successfully deleted export {export_name}"}
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        if total_time > 1.0:  # Log if total time exceeds 1 second
            print(f"Performance: Error in export deletion after {total_time:.3f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
