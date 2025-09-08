# routes/documents.py
import hashlib
import os

from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form,
                     HTTPException, UploadFile)
from sqlalchemy.orm import Session

from config import settings
from db.database import Document, SessionLocal, User, get_db
from logger_config import logger
from models.schemas import DocumentUpload
from services.pinecone_client import describe_index_stats
from services.redis_service import index_document

router = APIRouter()

@router.post("/upload", response_model=DocumentUpload)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    logger.info(f"Document upload requested by user {user_id}")
    
    # Check if file is None
    if file.filename is None:
        logger.warning("No file provided in document upload")
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Get user
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        logger.warning(f"User not found: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate file hash to prevent duplicates
    file_content = await file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    
    # Check if document already exists for this user
    existing_doc = db.query(Document).filter(
        Document.content_hash == file_hash,
        Document.user_id == user.id
    ).first()
    if existing_doc:
        logger.info(f"Document already exists for user {user_id}: {existing_doc.filename}")
        return DocumentUpload(
            filename=existing_doc.filename,
            status="already_exists"
        )
    
    # Save file
    os.makedirs(settings.assets_dir, exist_ok=True)
    file_path = os.path.join(settings.assets_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
    logger.info(f"Saved file to {file_path}")
    
    # Create document record with user_id
    document = Document(
        filename=file.filename,
        file_path=file_path,
        content_hash=file_hash,
        user_id=user.id
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    logger.info(f"Created document record with ID: {document.id}")
    
    # Create a new session for the background task
    new_db = SessionLocal()
    
    # Index document in background
    background_tasks.add_task(index_document, file_path, document.id, user.id, new_db)
    
    logger.info(f"Started background indexing task for document {document.id}")
    
    return DocumentUpload(
        filename=file.filename,
        status="indexing"
    )

@router.get("/list/{user_id}")
async def list_documents(user_id: str, db: Session = Depends(get_db)):
    """List all documents for a specific user."""
    logger.info(f"Listing documents for user {user_id}")
    
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        logger.warning(f"User not found: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    
    documents = db.query(Document).filter(Document.user_id == user.id).all()
    logger.info(f"Found {len(documents)} documents for user {user_id}")
    
    return {
        "user_id": user_id,
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "indexed": doc.indexed,
                "indexed_at": doc.indexed_at,
                "created_at": doc.created_at
            }
            for doc in documents
        ]
    }



@router.get("/test-embedding")
async def test_embedding():
    """Test embedding generation with a sample text."""
    logger.info("Testing embedding generation")
    
    try:
        # Import here to avoid circular imports
        from services.redis_service import get_embedding
        
        sample_text = "This is a test text to check if embedding generation works correctly."
        embedding = get_embedding(sample_text)
        
        logger.info(f"Generated embedding of length {len(embedding)}")
        
        return {
            "status": "success",
            "text": sample_text,
            "embedding_length": len(embedding),
            "first_5_values": embedding[:5]
        }
    except Exception as e:
        logger.error(f"Error testing embedding generation: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    

@router.get("/pinecone-stats")
async def get_pinecone_stats():
    """Get statistics about the Pinecone index."""
    logger.info("Getting Pinecone index statistics")
    
    try:
        # Import here to avoid circular imports
        
        stats = describe_index_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting Pinecone stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }