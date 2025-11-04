from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for documents (will use PostgreSQL in Week 2)
documents_db = []

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
    """Handle document upload"""
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
            os.remove(file_path)  # Clean up
            return jsonify({'error': 'Failed to extract text from document'}), 500
        
        # Calculate some basic stats
        word_count = len(text_content.split())
        char_count = len(text_content)
        
        # Store document metadata
        doc_metadata = {
            'id': len(documents_db) + 1,
            'filename': filename,
            'unique_filename': unique_filename,
            'file_path': file_path,
            'upload_date': datetime.now().isoformat(),
            'file_type': filename.rsplit('.', 1)[1].lower(),
            'word_count': word_count,
            'char_count': char_count,
            'text_content': text_content[:500] + '...' if len(text_content) > 500 else text_content  # Store preview
        }
        
        documents_db.append(doc_metadata)
        
        # Return response without full text content
        response_data = {k: v for k, v in doc_metadata.items() if k != 'text_content'}
        response_data['preview'] = text_content[:200] + '...' if len(text_content) > 200 else text_content
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'document': response_data
        }), 201
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of all uploaded documents"""
    try:
        # Return documents without full text content
        docs_list = []
        for doc in documents_db:
            doc_info = {k: v for k, v in doc.items() if k not in ['text_content', 'file_path']}
            docs_list.append(doc_info)
        
        return jsonify({'documents': docs_list}), 200
    except Exception as e:
        print(f"Error fetching documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document"""
    try:
        global documents_db
        
        # Find document
        doc = next((d for d in documents_db if d['id'] == doc_id), None)
        
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        # Delete file from disk
        if os.path.exists(doc['file_path']):
            os.remove(doc['file_path'])
        
        # Remove from database
        documents_db = [d for d in documents_db if d['id'] != doc_id]
        
        return jsonify({'message': 'Document deleted successfully'}), 200
        
    except Exception as e:
        print(f"Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat requests and stream responses from Anthropic API
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        if not ANTHROPIC_API_KEY:
            return jsonify({'error': 'API key not configured'}), 500

        # Prepare request to Anthropic API
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4096,
            'system': 'You are a helpful research assistant. Help users with their research questions, summarize information, explain concepts clearly, and assist with learning. Be concise but thorough.',
            'messages': messages,
            'stream': True
        }

        # Stream response from Anthropic
        def generate():
            with requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, stream=True) as response:
                if response.status_code != 200:
                    error_data = response.json()
                    yield f"data: {json.dumps({'error': error_data})}\n\n"
                    return
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            yield f"{decoded_line}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'api_key_configured': bool(ANTHROPIC_API_KEY),
        'documents_count': len(documents_db)
    })

if __name__ == '__main__':
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
    else:
        print("API key loaded successfully!")
    
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print("Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)