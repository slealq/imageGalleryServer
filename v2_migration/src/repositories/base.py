"""Base repository with common CRUD operations."""
from typing import Generic, TypeVar, Type, Optional, List, Any
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Base
from src.core.exceptions import NotFoundException

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository providing common database operations."""
    
    def __init__(self, model: Type[ModelType], db: AsyncSession):
        """
        Initialize repository.
        
        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
    
    async def get_by_id(self, id: UUID) -> Optional[ModelType]:
        """
        Get entity by ID.
        
        Args:
            id: Entity UUID
            
        Returns:
            Entity instance or None if not found
        """
        result = await self.db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_id_or_fail(self, id: UUID) -> ModelType:
        """
        Get entity by ID or raise exception.
        
        Args:
            id: Entity UUID
            
        Returns:
            Entity instance
            
        Raises:
            NotFoundException: If entity not found
        """
        entity = await self.get_by_id(id)
        if not entity:
            raise NotFoundException(
                resource=self.model.__tablename__,
                identifier=str(id)
            )
        return entity
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """
        Get all entities with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of entities
        """
        result = await self.db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())
    
    async def count(self) -> int:
        """
        Count total number of entities.
        
        Returns:
            Total count
        """
        result = await self.db.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()
    
    async def create(self, entity: ModelType) -> ModelType:
        """
        Create new entity.
        
        Args:
            entity: Entity instance to create
            
        Returns:
            Created entity with generated ID
        """
        self.db.add(entity)
        await self.db.flush()
        await self.db.refresh(entity)
        return entity
    
    async def update(self, id: UUID, values: dict) -> Optional[ModelType]:
        """
        Update entity by ID.
        
        Args:
            id: Entity UUID
            values: Dictionary of values to update
            
        Returns:
            Updated entity or None if not found
        """
        await self.db.execute(
            update(self.model)
            .where(self.model.id == id)
            .values(**values)
        )
        await self.db.flush()
        return await self.get_by_id(id)
    
    async def delete(self, id: UUID) -> bool:
        """
        Delete entity by ID.
        
        Args:
            id: Entity UUID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            delete(self.model).where(self.model.id == id)
        )
        await self.db.flush()
        return result.rowcount > 0
    
    async def exists(self, id: UUID) -> bool:
        """
        Check if entity exists.
        
        Args:
            id: Entity UUID
            
        Returns:
            True if exists, False otherwise
        """
        result = await self.db.execute(
            select(func.count()).select_from(self.model).where(self.model.id == id)
        )
        return result.scalar_one() > 0


