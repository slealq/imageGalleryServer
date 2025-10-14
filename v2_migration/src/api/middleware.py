"""API middleware for request timing, logging, etc."""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request timing and add request IDs."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with timing."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate timing
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        # Log slow requests
        if process_time > 1.0:
            print(
                f"SLOW REQUEST [{request_id}]: "
                f"{request.method} {request.url.path} "
                f"took {process_time:.3f}s"
            )
        
        return response

