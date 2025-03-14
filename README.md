# MediBot

A Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate responses to patient medical inquiries.

## Project Overview

MediBot leverages modern AI technologies to retrieve, analyze, and generate reliable medical information. The system combines web search capabilities with advanced language models to provide contextually relevant and accurate responses to medical questions.

## Technical Architecture

### Core Components

- **RAG Pipeline**: Implements a retrieval-augmented generation approach using LangChain
- **LLM**: Utilizes Google's Gemini 2.0 Flash model through a custom wrapper
- **Information Retrieval**: DuckDuckGo search API for fetching relevant medical information
- **Vector Store**: FAISS (Facebook AI Similarity Search) for efficient similarity search
- **Text Embedding**: HuggingFace's sentence-transformers/all-MiniLM-L6-v2 model
- **Text Processing**: RecursiveCharacterTextSplitter for chunking search results

### Workflow

1. User submits a medical query
2. System searches the web for relevant information using DuckDuckGo
3. Search results are split into manageable chunks
4. Text chunks are embedded into vector representations
5. Relevant information is retrieved based on vector similarity
6. Gemini LLM generates a comprehensive answer based on retrieved context
7. Response is returned with source attribution

## Setup and Installation

### Prerequisites

- Python 3.8+ (3.12 recomended)
- Google Gemini API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/OmBiradar/medibot.git
   cd medibot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   # Create .env file with your Gemini API key
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage
The main.py python script can be imported and used in other scripts as follows:
```python
# Basic usage example
from main import qa_chain

query = "What are the recommended treatments for type 2 diabetes?"
response = qa_chain.invoke({"query": query})

print("Answer:", response["result"])
print("Sources:", [doc.page_content for doc in response["source_documents"]])
```

## Features

- **Web-Enhanced Knowledge**: Augments LLM capabilities with real-time web search results
- **Source Attribution**: Provides sources for the information presented
- **Contextual Understanding**: Uses vector embeddings to find the most relevant information
- **Customizable**: Easily modify search parameters, chunk sizes, and retrieval settings

## Technical Implementation Details

- **Custom LLM Integration**: The `GeminiLLM` class wraps Google's Generative AI API to work seamlessly with LangChain
- **Optimized Chunking**: 1000-character chunks with 50-character overlap for better context preservation
- **Vector Retrieval**: Top-3 most relevant chunks retrieved for each query (k=3)
- **Environment Management**: API keys stored securely as environment variables

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Improvements

- Add medical knowledge base for specialized information
- Implement fact-checking against authoritative medical sources
- Add support for multi-modal inputs (images, documents)