import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.helpers.config import settings
from src.routes.api import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_name,
    description=settings.api_description,
    version="1.0.0"
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default route
@app.get("/", tags=['Health'])
async def root():
    return {
        "message": f"Welcome to the {settings.api_name}",
        "documentation": "/docs",
    }

# Include API routes
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    # Start the application
    logger.info(f"Starting {settings.api_name} on port {settings.api_port}")
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=settings.api_port, 
                reload=True)