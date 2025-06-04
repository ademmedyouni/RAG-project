# RAG-Powered Document Assistant with Voice Output

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that processes PDF documents, creates vector embeddings, and provides answers to user questions through both text and voice outputs. The system includes two main components:

1. A backend script (`final-RAG-voice.py`) that processes PDFs and implements the RAG pipeline with voice output
2. A Streamlit web interface (`pdf-rag-streamlit.py`) for interactive question answering

## Features

- **PDF Processing**: Extracts text from PDF documents in the `./data` directory
- **Text Chunking**: Splits documents into manageable chunks for better processing
- **Vector Embeddings**: Uses Ollama's embedding models to create semantic representations
- **Multi-Query Retrieval**: Generates multiple query variations to improve search results
- **LLM Integration**: Uses Llama3 for generating answers
- **Voice Output**: Converts text responses to speech using ElevenLabs API
- **Streamlit UI**: Provides a user-friendly web interface for interaction

## Installation

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   (Create a `requirements.txt` file with these packages if you don't have one:)
   ```
   ollama
   langchain
   langchain-community
   langchain-ollama
   pdfplumber
   unstructured
   chromadb
   streamlit
   elevenlabs
   ```

3. **Set up Ollama**:
   - Install Ollama from [ollama.ai](https://ollama.ai/)
   - Pull the required models:
     ```bash
     ollama pull llama3
     ollama pull nomic-embed-text
     ```

4. **Set up ElevenLabs (for voice output)**:
   - Sign up at [elevenlabs.io](https://elevenlabs.io/)
   - Get your API key and replace `"YOUR_API_KEY"` in `final-RAG-voice.py`

## Usage

### 1. Streamlit Web Interface

```bash
streamlit run pdf-rag-streamlit.py
```

[Insert screenshot of Streamlit UI here]

The interface will open in your default browser. You can:
- Enter questions in the text input field
- View generated responses
- The system automatically processes PDFs in the background

### 2. Command Line with Voice Output

```bash
python final-RAG-voice.py
```

This script will:
1. Process all PDFs in the `./data` directory
2. Create vector embeddings
3. Answer a predefined question
4. Generate a voice output of the answer as an MP3 file

## File Structure

```
project-root/
│
├── data/                   # Directory for PDF documents
│   └── BOI.pdf             # Example PDF document
│
├── db/                     # Vector database storage
│   └── vector_db/          # ChromaDB persistence directory
│
├── chroma_db/              # Alternative vector DB storage
│
├── final-RAG-voice.py      # Main RAG pipeline with voice output
├── pdf-rag-streamlit.py    # Streamlit web interface
└── README.md               # This file
```

## Configuration

You can customize these aspects of the system:

1. **Models**: Change `MODEL_NAME` or `EMBEDDING_MODEL` in the scripts
2. **Chunking**: Adjust `chunk_size` and `chunk_overlap` in the text splitter
3. **Voice**: Modify voice parameters in the `text_to_speech_file` function
4. **PDF Directory**: Change the `./data` path to point to your documents

## Troubleshooting

- **Missing PDFs**: Ensure your documents are in the `./data` directory
- **Ollama Errors**: Verify models are downloaded (`ollama list`)
- **API Key Issues**: Check your ElevenLabs API key is correctly set
- **Vector DB Issues**: Delete the `./db` or `./chroma_db` directories to rebuild

## Future Enhancements

- Add support for more document types (Word, HTML, etc.)
- Implement document upload through the Streamlit interface
- Add conversation history
- Provide voice input option
- Implement user authentication for sensitive documents


## Screenshots

1. The Streamlit interface with a sample question and response
![Capture d'écran 2025-06-04 081612](https://github.com/user-attachments/assets/35e09c39-cadf-41b3-b814-ce05bac65359)
  ![Capture d'écran 2025-06-04 081836](https://github.com/user-attachments/assets/677ecf5e-932e-47da-9dbb-77d3ac222658)

2. The command-line output during PDF processing:
![Capture d'écran 2025-06-04 113053](https://github.com/user-attachments/assets/31525791-f8e0-4dea-8c25-96cbaa54b1d8)


