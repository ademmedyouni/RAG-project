
# Document Assistant with RAG and Voice Integration

**Author**: Adem Medyouni  
**Technologies**: Python, LangChain, Ollama, ChromaDB, Streamlit, ElevenLabs  

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides conversational access to their content through both web and voice interfaces. The system features:

- Document ingestion and vector embedding
- Multi-query retrieval for improved accuracy
- Local LLM processing via Ollama
- Two interface options: Streamlit web UI and voice-enabled CLI

## Features

### Core Capabilities
- 📄 Process multiple PDF documents simultaneously
- 🔍 Semantic search with context-aware retrieval
- 💬 Natural language question answering
- 🗣️ Voice response generation (ElevenLabs integration)

### Technical Highlights
- Local LLM processing with Llama3
- ChromaDB vector storage with persistent database
- Advanced chunking with metadata management
- Multi-query retrieval for improved relevance

## Installation

### Prerequisites
- Python 3.9+
- Ollama installed and running ([installation guide](https://ollama.ai/))
- ElevenLabs API key (for voice version)

### Setup
```bash
git clone https://github.com/adem-emdyouni/document-assistant.git
cd document-assistant
pip install -r requirements.txt

# Pull required Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Usage

### Streamlit Web Interface
```bash
streamlit run pdf-rag-streamlit.py
```

### Voice-Enabled Version
```bash
python final-rag-voice.py
```

## Configuration

1. Place your PDF documents in the `./data` directory
2. For voice version, create a `.env` file with:
```env
ELEVENLABS_API_KEY=your_api_key_here
```

## Project Structure
```
document-assistant/
├── data/                  # PDF documents storage
├── chroma_db/             # Vector database storage (auto-created)
├── pdf-rag-streamlit.py   # Web interface implementation
├── final-rag-voice.py     # Voice interface implementation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Performance Notes

- First run will be slower as it processes documents and creates embeddings
- Subsequent queries are faster using the persisted vector store
- For best voice performance, use ElevenLabs Turbo v2 model
