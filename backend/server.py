from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

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
        'api_key_configured': bool(ANTHROPIC_API_KEY)
    })

if __name__ == '__main__':
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
    else:
        print("API key loaded successfully!")
    
    print("Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)