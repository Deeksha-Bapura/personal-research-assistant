# Personal Research Assistant 🔍

An AI-powered research assistant built with Claude (Anthropic) to help with research tasks, learning, and knowledge management.

## 🎯 Project Goals

**Phase 1 (Current):** Basic LLM chat interface with conversation history
- ✅ Clean chat interface
- ✅ Streaming responses
- ✅ Conversation memory
- ✅ API key management

**Phase 2 (Coming Soon):** RAG Integration
- 📄 Document upload (PDF, TXT, DOCX, MD)
- 🔍 Semantic search across documents
- 📚 Chat with your knowledge base
- 🎯 Source citations

**Phase 3 (Future):** Advanced Features
- 📁 Document collections
- 📊 Usage analytics
- 🔄 Multiple LLM support
- 📤 Export conversations

## 🚀 Quick Start

### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, Edge)
- An Anthropic API key ([Get one here](https://console.anthropic.com/))

### Setup

1. **Clone or download this project**
   ```bash
   cd personal-research-assistant
   ```

2. **Open `index.html` in your browser**
   - Just double-click the file, or
   - Right-click → Open with → Your browser

3. **Enter your API key**
   - The app will prompt you for your Anthropic API key
   - Your key is stored locally in your browser (never sent anywhere except Anthropic's API)

4. **Start chatting!**
   - Ask questions
   - Get research help
   - Explore concepts

## 📁 Project Structure

```
personal-research-assistant/
├── frontend/
│   └── index.html          # React web interface
├── backend/
│   ├── server.py           # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── .env               # API keys (not in git)
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## 🛠️ Tech Stack

**Current (Phase 1):**
- React 18 (via CDN)
- Tailwind CSS (via CDN)
- Anthropic Claude API (claude-sonnet-4)
- Vanilla JavaScript

**Planned (Phase 2+):**
- Backend: Python (FastAPI) or Node.js (Express)
- Vector Database: Chroma, Pinecone, or pgvector
- Document Processing: PyPDF2, python-docx, Unstructured
- RAG Framework: LangChain or LlamaIndex
- Database: PostgreSQL

## 💡 Usage Tips

- **Press Enter** to send a message
- **Shift + Enter** for a new line
- Use "Clear Chat" to start fresh
- Your API key persists across sessions
- Conversations are stored in your browser (localStorage)

## 🔐 Privacy & Security

- Your API key is stored in browser localStorage (client-side only)
- Conversations are stored locally in your browser
- No data is sent to any server except Anthropic's API
- Clear your browser data to remove all stored information

## 🎓 Learning Resources

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Claude Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [RAG Concepts](https://www.anthropic.com/research)

## 🗺️ Roadmap

- [x] Phase 1: Basic LLM chat interface
- [ ] Phase 2: RAG implementation with document upload
- [ ] Phase 3: Advanced features and optimization
- [ ] Phase 4: Deployment and production readiness

## 📝 Notes

This is currently a single-file HTML application for rapid prototyping. As we add RAG capabilities, we'll refactor into a proper full-stack application with a backend.

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with the code
- Add new features
- Optimize performance
- Share your improvements

## 📄 License

This project is for educational purposes. Use at your own discretion.
