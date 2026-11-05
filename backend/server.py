from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json
import itertools
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from datetime import datetime
import chromadb
from rag import RAGConfig, Retriever, build_system_prompt, stream_claude
load_dotenv()

app = Flask(__name__)
CORS(app)

from agent import agent_bp
app.register_blueprint(agent_bp)

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Chunking / retrieval settings live in rag.py (RAGConfig) so the eval suite
# measures exactly what this server runs. Override with RAG_* env vars.
RAG_CONFIG = RAGConfig.from_env()

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('chroma_db', exist_ok=True)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Embedding model (and reranker, if enabled) load once on startup
print(f"Loading embedding model {RAG_CONFIG.embedding_model}...")
retriever = Retriever(RAG_CONFIG, collection=collection)
print("Embedding model loaded!")

# In-memory storage for documents
documents_db = []


def _first_free_doc_id():
    """One past the highest doc_id already stored in Chroma.

    The document list above is in memory and resets on restart, but Chroma
    keeps its chunks, so ids must continue after anything already stored.
    """
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []
    return max((int(m["doc_id"]) for m in metadatas if m and "doc_id" in m), default=0) + 1


# Document ids only ever go up, so an id is never reused after a delete
# (issue #3). next() on itertools.count is atomic in CPython, so concurrent
# uploads still get distinct ids.
_doc_ids = itertools.count(_first_free_doc_id())

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        return None

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting DOCX text: {str(e)}")
        return None

def extract_text_from_txt(file_path):
    """Extract text from TXT/MD file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading text file: {str(e)}")
        return None

def process_document(file_path, filename):
    """Process document and extract text based on file type"""
    extension = filename.rsplit('.', 1)[1].lower()
    
    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)
    elif extension in ['txt', 'md']:
        return extract_text_from_txt(file_path)
    else:
        return None

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Handle document upload and create embeddings"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Extract text from document
        text_content = process_document(file_path, filename)
        
        if text_content is None:
            os.remove(file_path)
            return jsonify({'error': 'Failed to extract text from document'}), 500
        
        # Chunk, embed and store in Chroma (see rag.Retriever)
        doc_id = next(_doc_ids)
        chunks = retriever.add_document(doc_id, filename, text_content)
        
        if not chunks:
            os.remove(file_path)
            return jsonify({'error': 'Document is empty or could not be chunked'}), 400
        
        # Calculate stats
        word_count = len(text_content.split())
        char_count = len(text_content)
        
        # Store document metadata
        doc_metadata = {
            'id': doc_id,
            'filename': filename,
            'unique_filename': unique_filename,
            'file_path': file_path,
            'upload_date': datetime.now().isoformat(),
            'file_type': filename.rsplit('.', 1)[1].lower(),
            'word_count': word_count,
            'char_count': char_count,
            'chunk_count': len(chunks),
            'text_preview': text_content[:500] + '...' if len(text_content) > 500 else text_content
        }
        
        documents_db.append(doc_metadata)
        
        # Return response
        response_data = {k: v for k, v in doc_metadata.items() if k != 'text_preview'}
        response_data['preview'] = text_content[:200] + '...' if len(text_content) > 200 else text_content
        
        return jsonify({
            'message': 'Document uploaded and indexed successfully',
            'document': response_data
        }), 201
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of all uploaded documents"""
    try:
        docs_list = []
        for doc in documents_db:
            doc_info = {k: v for k, v in doc.items() if k not in ['text_preview', 'file_path']}
            docs_list.append(doc_info)
        
        return jsonify({'documents': docs_list}), 200
    except Exception as e:
        print(f"Error fetching documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and its embeddings"""
    try:
        global documents_db
        
        # Find document
        doc = next((d for d in documents_db if d['id'] == doc_id), None)
        
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        # Delete embeddings from Chroma
        # Get all chunk IDs for this document
        chunk_ids = [f"doc_{doc_id}_chunk_{i}" for i in range(doc['chunk_count'])]
        try:
            collection.delete(ids=chunk_ids)
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
        
        # Delete file from disk
        if os.path.exists(doc['file_path']):
            os.remove(doc['file_path'])
        
        # Remove from database
        documents_db = [d for d in documents_db if d['id'] != doc_id]
        
        return jsonify({'message': 'Document deleted successfully'}), 200
        
    except Exception as e:
        print(f"Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Search documents using semantic similarity"""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 3)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        search_results = retriever.search(query, top_k=top_k)
        
        return jsonify({
            'query': query,
            'results': search_results
        }), 200
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with RAG - retrieve relevant context from documents"""
    try:
        data = request.json
        messages = data.get('messages', [])
        use_rag = data.get('use_rag', True)  # Enable RAG by default
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        if not ANTHROPIC_API_KEY:
            return jsonify({'error': 'API key not configured'}), 500

        # Get the last user message for context retrieval
        last_user_message = next((m for m in reversed(messages) if m['role'] == 'user'), None)
        
        # Retrieve relevant chunks (if RAG is on and documents exist) and
        # build the system prompt - same code path the eval suite scores
        hits = []
        if use_rag and last_user_message and len(documents_db) > 0:
            hits = retriever.search(last_user_message['content'])
        system_prompt = build_system_prompt(hits)

        # Stream response from Anthropic
        def generate():
            for line in stream_claude(system_prompt, messages, RAG_CONFIG, ANTHROPIC_API_KEY):
                yield f"{line}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'api_key_configured': bool(ANTHROPIC_API_KEY),
        'documents_count': len(documents_db),
        'embeddings_count': collection.count(),
        'embedding_model': RAG_CONFIG.embedding_model,
        'rag_config': RAG_CONFIG.to_dict()
    })

if __name__ == '__main__':
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
    else:
        print("✓ API key loaded successfully!")
    
    print(f"✓ Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"✓ Chroma DB: {os.path.abspath('chroma_db')}")
    print(f"✓ RAG config: {RAG_CONFIG.to_dict()}")
    print(f"✓ Documents in DB: {len(documents_db)}")
    print(f"✓ Vector embeddings: {collection.count()}")
    print("\nStarting server on http://localhost:5000")
    print("RAG is ENABLED - uploaded documents will be searchable!\n")
    app.run(debug=True, port=5000)