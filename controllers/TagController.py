from fastapi import APIRouter, HTTPException
from models.models import TagResponse, AddTagRequest
from config import IMAGES_DIR
from filter_manager import filter_manager

router = APIRouter(prefix="/images", tags=["tags"])

@router.get("/{image_id}/tags", response_model=TagResponse)
async def get_image_tags(image_id: str):
    """
    Get all tags for an image (both scene and image-specific tags)
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get all tags for the image
        tags = filter_manager.get_tags_for_image(image_id)
        
        return TagResponse(tags=tags)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{image_id}/custom-tags", response_model=TagResponse)
async def get_image_custom_tags(image_id: str):
    """
    Get only the custom tags for an image (tags that were added specifically to this image)
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get only the custom tags for the image
        tags = filter_manager.get_tags_from_file_for_image(image_id)
        
        return TagResponse(tags=tags)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{image_id}/tags")
async def add_image_tag(image_id: str, request: AddTagRequest):
    """
    Add a tag to an image
    """
    try:
        # Verify image exists
        image_files = list(IMAGES_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Add the tag
        filter_manager.set_tags_in_file_for_image(image_id, [request.tag])
        
        return {"message": f"Tag '{request.tag}' added successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 