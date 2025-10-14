"""Storage service for file operations."""
import shutil
from pathlib import Path
from typing import Optional, BinaryIO
from uuid import UUID

from src.core.config import settings
from src.core.exceptions import StorageException


class StorageService:
    """Service for managing file storage operations."""
    
    def __init__(self):
        """Initialize storage service with configured paths."""
        self.images_dir = settings.images_dir
        self.thumbnails_dir = settings.thumbnails_dir
        self.crops_dir = settings.crops_dir
        self.archives_dir = settings.archives_dir
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all storage directories exist."""
        for directory in [
            self.images_dir,
            self.thumbnails_dir,
            self.crops_dir,
            self.archives_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_image_path(self, photoset_id: UUID, filename: str) -> Path:
        """
        Get the full path for an image file.
        
        Args:
            photoset_id: Photoset UUID
            filename: Image filename
            
        Returns:
            Full path to image file
        """
        photoset_dir = self.images_dir / str(photoset_id)
        photoset_dir.mkdir(parents=True, exist_ok=True)
        return photoset_dir / filename
    
    def get_thumbnail_path(self, image_id: UUID) -> Path:
        """
        Get the full path for a thumbnail file.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Full path to thumbnail file
        """
        return self.thumbnails_dir / f"{image_id}.jpg"
    
    def get_crop_path(self, image_id: UUID) -> Path:
        """
        Get the full path for a crop file.
        
        Args:
            image_id: Image UUID
            
        Returns:
            Full path to crop file
        """
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        return self.crops_dir / f"{image_id}.png"
    
    def save_file(self, file_data: bytes, file_path: Path) -> Path:
        """
        Save binary data to a file.
        
        Args:
            file_data: Binary file data
            file_path: Destination path
            
        Returns:
            Path to saved file
            
        Raises:
            StorageException: If save fails
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            return file_path
        except Exception as e:
            raise StorageException(f"Failed to save file {file_path}: {e}")
    
    def save_upload(self, upload_file: BinaryIO, file_path: Path) -> Path:
        """
        Save an uploaded file.
        
        Args:
            upload_file: Uploaded file object
            file_path: Destination path
            
        Returns:
            Path to saved file
            
        Raises:
            StorageException: If save fails
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(upload_file, f)
            return file_path
        except Exception as e:
            raise StorageException(f"Failed to save upload {file_path}: {e}")
    
    def read_file(self, file_path: Path) -> bytes:
        """
        Read binary data from a file.
        
        Args:
            file_path: File path to read
            
        Returns:
            Binary file data
            
        Raises:
            StorageException: If read fails
        """
        try:
            if not file_path.exists():
                raise StorageException(f"File not found: {file_path}")
            
            with open(file_path, 'rb') as f:
                return f.read()
        except StorageException:
            raise
        except Exception as e:
            raise StorageException(f"Failed to read file {file_path}: {e}")
    
    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: File path to delete
            
        Returns:
            True if deleted, False if file didn't exist
            
        Raises:
            StorageException: If deletion fails
        """
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            raise StorageException(f"Failed to delete file {file_path}: {e}")
    
    def delete_directory(self, dir_path: Path) -> bool:
        """
        Delete a directory and all its contents.
        
        Args:
            dir_path: Directory path to delete
            
        Returns:
            True if deleted, False if directory didn't exist
            
        Raises:
            StorageException: If deletion fails
        """
        try:
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)
                return True
            return False
        except Exception as e:
            raise StorageException(f"Failed to delete directory {dir_path}: {e}")
    
    def file_exists(self, file_path: Path) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        return file_path.exists() and file_path.is_file()
    
    def get_file_size(self, file_path: Path) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: File path
            
        Returns:
            File size in bytes
            
        Raises:
            StorageException: If file doesn't exist
        """
        if not file_path.exists():
            raise StorageException(f"File not found: {file_path}")
        
        return file_path.stat().st_size
    
    def cleanup_orphaned_files(self, valid_paths: set[Path], directory: Path):
        """
        Delete files in directory that aren't in the valid paths set.
        
        Args:
            valid_paths: Set of paths that should be kept
            directory: Directory to clean
        """
        if not directory.exists():
            return
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path not in valid_paths:
                try:
                    file_path.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup

