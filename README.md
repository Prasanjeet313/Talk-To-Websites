# Webpage RAG Chatbot 🤖

A powerful RAG (Retrieval-Augmented Generation) chatbot that allows you to chat with any webpage using AI. Built with Streamlit, LangChain, LangGraph, FAISS, and Groq.

## Features ✨

- **Web Scraping**: Automatically scrape and process any webpage
- **Advanced RAG**: Uses FAISS vector database for efficient retrieval
- **GPU Acceleration**: Supports CUDA for faster embeddings
- **Conversational Memory**: Maintains chat history for context-aware responses
- **LangGraph Workflow**: Organized processing pipeline using LangGraph
- **Beautiful UI**: Clean Streamlit interface for easy interaction
- **Source Attribution**: Shows relevant sources for each answer

## Architecture 🏗️

The application is built with a clean OOP structure:

```
├── WebScraper: Handles webpage scraping
├── DocumentProcessor: Chunks and processes documents
├── VectorStoreManager: Manages FAISS vector store and embeddings
├── RAGChatbot: Main chatbot using LangChain and Groq
└── WebpageRAGWorkflow: LangGraph workflow orchestration
```

### Workflow Pipeline

```
URL Input → Scrape → Process & Chunk → Vectorize → Ready for Q&A
```

## Prerequisites 📋

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- Groq API Key

## Installation 🚀

### 1. Clone or download the project

```bash
# Create a new directory
mkdir webpage-rag-chatbot
cd webpage-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**For GPU Support (CUDA):**
```bash
# Replace faiss-cpu with faiss-gpu in requirements.txt
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 3. Configure environment variables

Edit the `.env` file with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Usage 💻

### Run the application

```bash
streamlit run webpage_rag_chatbot_simple.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Chatbot

1. **Enter URL**: In the sidebar, enter the webpage URL you want to chat with
   - Example: `https://www.emiratesnbd.com/en/cards/credit-cards`

2. **Load Webpage**: Click the "Load Webpage" button
   - The app will scrape, process, and vectorize the content
   - You'll see a success message when ready

3. **Ask Questions**: Start chatting with the webpage!
   - Example questions:
     - "What credit cards are available?"
     - "What are the benefits of the Platinum card?"
     - "What is the annual fee for the gold card?"
     - "How can I apply for a credit card?"

4. **View Sources**: Click on "View Sources" to see the relevant chunks used for the answer

5. **Clear Chat**: Use the "Clear Chat History" button to start a new conversation

## Configuration ⚙️

### Groq Models

You can change the Groq model in the `Config` class:

```python
GROQ_MODEL: str = "gemma-7b-it"  # Default
# Other options:
# - "llama2-70b-4096"
# - "gemma-7b-it"mixtral-8x7b-32768
# - "llama3-70b-8192"
```

### Embedding Models

Change the embedding model for different performance/quality tradeoffs:

```python
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, good quality
# Other options:
# - "sentence-transformers/all-mpnet-base-v2"  # Better quality, slower
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
```

### Chunk Settings

Adjust chunking parameters for different webpage types:

```python
CHUNK_SIZE: int = 1000        # Size of each text chunk
CHUNK_OVERLAP: int = 200      # Overlap between chunks
```

## How It Works 🔍

### 1. Web Scraping
- Uses BeautifulSoup to extract clean content from webpages
- Removes unnecessary elements (nav, footer, scripts, styles)
- Preserves main content structure

### 2. Document Processing
- Splits content into manageable chunks using RecursiveCharacterTextSplitter
- Maintains context with overlapping chunks
- Preserves metadata (URL, title)

### 3. Vector Store Creation
- Embeds document chunks using HuggingFace embeddings
- Stores embeddings in FAISS for efficient similarity search
- Supports GPU acceleration with CUDA

### 4. RAG Pipeline
- Uses ConversationalRetrievalChain for context-aware responses
- Retrieves relevant chunks based on question similarity
- Generates answers using Groq LLM with retrieved context
- Maintains conversation memory for follow-up questions

### 5. LangGraph Workflow
- Orchestrates the entire pipeline using state machine
- Provides error handling and status tracking
- Enables extensibility for future features

## Code Structure 📁

```python
# Main Classes:

Config                    # Configuration dataclass
WebScraper               # Web scraping functionality
DocumentProcessor        # Document chunking and processing
VectorStoreManager       # FAISS vector store management
RAGChatbot              # Conversational RAG chain
WebpageRAGWorkflow      # LangGraph workflow orchestration
```

## Troubleshooting 🔧

### Common Issues

**1. CUDA Out of Memory**
- Reduce `CHUNK_SIZE` in Config
- Use CPU instead: Change `device='cuda'` to `device='cpu'` in VectorStoreManager

**2. Slow Embedding Generation**
- First run downloads the embedding model (~80MB)
- Subsequent runs will be faster
- Consider using GPU if available

**3. Webpage Scraping Fails**
- Some websites block scrapers
- Try different websites
- Check if the URL is accessible

**4. Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## Advanced Usage 🎓

### Custom Prompts

Modify the prompt template in `RAGChatbot.create_qa_chain()`:

```python
template = """Your custom prompt here...

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:"""
```

### Adding More Retrievers

Modify the retriever settings in `VectorStoreManager.get_retriever()`:

```python
return self.vector_store.as_retriever(
    search_type="mmr",  # or "similarity"
    search_kwargs={"k": 6, "fetch_k": 10}
)
```

## Performance Tips 🚀

1. **Use GPU**: Install `faiss-gpu` for 5-10x faster embedding
2. **Optimize Chunks**: Smaller chunks = faster search, but may lose context
3. **Cache Embeddings**: Vector stores can be saved and reloaded
4. **Batch Processing**: Process multiple URLs in parallel (future feature)

## Limitations ⚠️

- JavaScript-heavy websites may not scrape properly
- Very large webpages may take time to process
- Rate limiting may apply for the Groq API
- Dynamic content may not be captured

## Future Enhancements 🔮

- [ ] Support for multiple URLs
- [ ] PDF and document upload support
- [ ] Export chat history
- [ ] Save and load vector stores
- [ ] Advanced filtering and search options
- [ ] Multi-language support
- [ ] Authentication for restricted pages

## Contributing 🤝

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## License 📄

This project is open source and available under the MIT License.

## Acknowledgments 🙏

- LangChain for the RAG framework
- Groq for the fast LLM API
- Streamlit for the beautiful UI
- HuggingFace for embeddings
- FAISS for vector search

---

**Built with ❤️ using LangChain, LangGraph, and Streamlit**
