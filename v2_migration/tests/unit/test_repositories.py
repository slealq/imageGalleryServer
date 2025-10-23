"""Unit tests for repositories."""
from __future__ import annotations

import pytest

from src.repositories import PhotosetRepository, ImageRepository, CaptionRepository
from tests.factories import PhotosetFactory, ImageFactory, CaptionFactory


@pytest.mark.asyncio
class TestPhotosetRepository:
    """Test PhotosetRepository."""
    
    async def test_create_photoset(self, db_session):
        """Test creating a photoset."""
        repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create()
        
        created = await repo.create(photoset)
        await db_session.commit()
        
        assert created.id == photoset.id
        assert created.name == photoset.name
        assert created.year == photoset.year
    
    async def test_get_photoset_by_id(self, db_session):
        """Test getting a photoset by ID."""
        repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create()
        
        await repo.create(photoset)
        await db_session.commit()
        
        found = await repo.get_by_id(photoset.id)
        
        assert found is not None
        assert found.id == photoset.id
        assert found.name == photoset.name
    
    async def test_update_photoset(self, db_session):
        """Test updating a photoset."""
        repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create(name="Original Name")
        
        await repo.create(photoset)
        await db_session.commit()
        
        updated = await repo.update(photoset.id, {"name": "Updated Name"})
        await db_session.commit()
        
        assert updated.name == "Updated Name"
    
    async def test_delete_photoset(self, db_session):
        """Test deleting a photoset."""
        repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create()
        
        await repo.create(photoset)
        await db_session.commit()
        
        deleted = await repo.delete(photoset.id)
        await db_session.commit()
        
        assert deleted is True
        
        found = await repo.get_by_id(photoset.id)
        assert found is None


@pytest.mark.asyncio
class TestImageRepository:
    """Test ImageRepository."""
    
    async def test_create_image(self, db_session):
        """Test creating an image."""
        # First create a photoset
        photoset_repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create()
        await photoset_repo.create(photoset)
        await db_session.commit()
        
        # Create image
        repo = ImageRepository(db_session)
        image = ImageFactory.create(photoset_id=photoset.id)
        
        created = await repo.create(image)
        await db_session.commit()
        
        assert created.id == image.id
        assert created.original_filename == image.original_filename
        assert created.photoset_id == photoset.id
    
    async def test_get_images_by_photoset(self, db_session):
        """Test getting images by photoset."""
        # Create photoset
        photoset_repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create()
        await photoset_repo.create(photoset)
        
        # Create images
        repo = ImageRepository(db_session)
        image1 = ImageFactory.create(photoset_id=photoset.id, original_filename="img1.jpg")
        image2 = ImageFactory.create(photoset_id=photoset.id, original_filename="img2.jpg")
        
        await repo.create(image1)
        await repo.create(image2)
        await db_session.commit()
        
        # Get images
        images = await repo.get_by_photoset(photoset.id)
        
        assert len(images) == 2
        assert {img.original_filename for img in images} == {"img1.jpg", "img2.jpg"}


@pytest.mark.asyncio
class TestCaptionRepository:
    """Test CaptionRepository."""
    
    async def test_upsert_caption_create(self, db_session):
        """Test upserting a caption (create)."""
        # Create image
        photoset_repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create()
        await photoset_repo.create(photoset)
        
        image_repo = ImageRepository(db_session)
        image = ImageFactory.create(photoset_id=photoset.id)
        await image_repo.create(image)
        await db_session.commit()
        
        # Create caption
        repo = CaptionRepository(db_session)
        caption = await repo.upsert(
            image_id=image.id,
            caption_text="Test caption",
            generator_type="test"
        )
        await db_session.commit()
        
        assert caption.image_id == image.id
        assert caption.caption == "Test caption"
    
    async def test_upsert_caption_update(self, db_session):
        """Test upserting a caption (update)."""
        # Create image and initial caption
        photoset_repo = PhotosetRepository(db_session)
        photoset = PhotosetFactory.create()
        await photoset_repo.create(photoset)
        
        image_repo = ImageRepository(db_session)
        image = ImageFactory.create(photoset_id=photoset.id)
        await image_repo.create(image)
        
        repo = CaptionRepository(db_session)
        caption = await repo.upsert(
            image_id=image.id,
            caption_text="Original caption",
            generator_type="test"
        )
        await db_session.commit()
        
        # Update caption
        updated = await repo.upsert(
            image_id=image.id,
            caption_text="Updated caption",
            generator_type="test"
        )
        await db_session.commit()
        
        assert updated.id == caption.id  # Same record
        assert updated.caption == "Updated caption"





