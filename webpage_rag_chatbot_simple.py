"""
Webpage RAG Chatbot - Streamlit Cloud Compatible Version
Uses LangChain, FAISS, and Groq LLM (Pure Python Scraping)
"""

import os
import streamlit as st
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import logging
import time

from dotenv import load_dotenv
# Load env vars
load_dotenv()

# --- MODIFIED IMPORTS FOR STABILITY ---
# Use langchain_text_splitters to avoid import errors in newer versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the application"""
    # Use st.secrets for Cloud deployment compatibility, fallback to os.environ
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

class WebScraper:
    """
    Streamlit Cloud Compatible Scraper
    Uses requests + BeautifulSoup instead of Selenium to avoid browser binary issues.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a given URL using requests
        """
        try:
            logger.info(f"Scraping URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
                script.decompose()

            # Get title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else url

            # Get main content
            # Try to find specific content containers first
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Clean up text (remove excessive newlines)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            if len(text) < 100:
                logger.warning("Scraped content is very short. Page might be JS-rendered.")

            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            return {
                'url': url,
                'title': title_text,
                'content': text,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': str(e)
            }

class DocumentProcessor:
    """Handles document chunking and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_scraped_data(self, scraped_data: Dict[str, Any]) -> List[Document]:
        if not scraped_data.get('success'):
            return []
        
        metadata = {
            'source': scraped_data['url'],
            'title': scraped_data['title']
        }
        
        document = Document(
            page_content=scraped_data['content'],
            metadata=metadata
        )
        
        chunks = self.text_splitter.split_documents([document])
        return chunks

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self, embedding_model: str, use_gpu: bool = False):
        # Force CPU for Streamlit Cloud to avoid memory/compatibility issues
        device = 'cpu'
        logger.info(f"Initializing embeddings on {device}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store: Optional[FAISS] = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("No documents provided")
        
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vector_store
    
    def get_retriever(self, k: int = 4):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

class RAGChatbot:
    """Main RAG chatbot using LangChain and Groq"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Check API Key
        if not config.GROQ_API_KEY:
            st.error("Groq API Key not found! Please add it to secrets or .env")
            st.stop()

        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        
    def create_qa_chain(self, retriever):
        template = """You are a helpful AI assistant.
Use the following pieces of context from the webpage to answer the question at the end.
If the answer isn't in the context, say you don't know.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return self.qa_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {'success': False, 'answer': "Please load a webpage first."}
            
        try:
            result = self.qa_chain({"question": question})
            return {
                'answer': result['answer'],
                'source_documents': result.get('source_documents', []),
                'success': True
            }
        except Exception as e:
            return {'success': False, 'answer': f"Error: {str(e)}"}
    
    def clear_memory(self):
        self.memory.clear()

class WebpageRAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.scraper = WebScraper()
        self.processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
        self.chatbot = RAGChatbot(config)
    
    def process_url(self, url: str) -> Dict[str, Any]:
        try:
            scraped_data = self.scraper.scrape_url(url)
            if not scraped_data['success']:
                return {'success': False, 'error': scraped_data.get('error')}
            
            documents = self.processor.process_scraped_data(scraped_data)
            if not documents:
                return {'success': False, 'error': 'No content found on page'}
            
            self.vector_manager.create_vector_store(documents)
            retriever = self.vector_manager.get_retriever()
            self.chatbot.create_qa_chain(retriever)
            
            return {
                'success': True,
                'title': scraped_data['title'],
                'num_chunks': len(documents),
                'content_length': len(scraped_data['content'])
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def ask(self, question: str):
        return self.chatbot.ask(question)
    
    def clear_memory(self):
        self.chatbot.clear_memory()

# Streamlit UI
def main():
    st.set_page_config(page_title="Webpage RAG Chatbot", page_icon="🤖")
    st.title("🤖 Webpage Chatbot")
    
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    with st.sidebar:
        st.header("Configuration")
        url = st.text_input("Webpage URL", "https://www.example.com")
        
        if st.button("Load Webpage", type="primary"):
            with st.spinner("Processing..."):
                pipeline = WebpageRAGPipeline(st.session_state.config)
                result = pipeline.process_url(url)
                
                if result['success']:
                    st.session_state.pipeline = pipeline
                    st.session_state.chat_history = []
                    st.success(f"Loaded: {result['title']}")
                    st.info(f"Chunks: {result['num_chunks']}")
                else:
                    st.error(f"Error: {result.get('error')}")

        if st.button("Clear History"):
            st.session_state.chat_history = []
            if st.session_state.pipeline:
                st.session_state.pipeline.clear_memory()
            st.rerun()

    # Chat Interface
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for doc in msg["sources"][:2]:
                    st.caption(doc.page_content[:300] + "...")

    if prompt := st.chat_input():
        if not st.session_state.pipeline:
            st.error("Please load a webpage first!")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.pipeline.ask(prompt)
                    st.write(response['answer'])
                    
                    if response.get('source_documents'):
                        with st.expander("Sources"):
                            for doc in response['source_documents'][:2]:
                                st.caption(doc.page_content[:300] + "...")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "sources": response.get('source_documents')
                    })

if __name__ == "__main__":
    main()