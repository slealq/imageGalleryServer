"""Custom exceptions for the application."""


class GalleryException(Exception):
    """Base exception for all gallery-related errors."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class NotFoundException(GalleryException):
    """Raised when a resource is not found."""
    
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} with identifier '{identifier}' not found"
        super().__init__(message, status_code=404)


class ValidationException(GalleryException):
    """Raised when validation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class DuplicateException(GalleryException):
    """Raised when attempting to create a duplicate resource."""
    
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} with identifier '{identifier}' already exists"
        super().__init__(message, status_code=409)


class StorageException(GalleryException):
    """Raised when file storage operations fail."""
    
    def __init__(self, message: str):
        super().__init__(f"Storage error: {message}", status_code=500)


class CacheException(GalleryException):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str):
        super().__init__(f"Cache error: {message}", status_code=500)


class DatabaseException(GalleryException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str):
        super().__init__(f"Database error: {message}", status_code=500)


class CaptionGenerationException(GalleryException):
    """Raised when caption generation fails."""
    
    def __init__(self, message: str):
        super().__init__(f"Caption generation error: {message}", status_code=500)


class ImageProcessingException(GalleryException):
    """Raised when image processing fails."""
    
    def __init__(self, message: str):
        super().__init__(f"Image processing error: {message}", status_code=500)


