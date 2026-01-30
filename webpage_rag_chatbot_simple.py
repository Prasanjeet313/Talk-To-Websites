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
import json
import hashlib
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime

from dotenv import load_dotenv
# Load env vars
load_dotenv()

# Optional Cloudflare-bypass libraries
try:
    import curl_cffi.requests as cc_requests
    CURL_CFFI_AVAILABLE = True
except Exception:
    cc_requests = None
    CURL_CFFI_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

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
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "outputs")
    # scraper backend: 'auto' | 'requests' | 'curl_cffi' | 'playwright'
    SCRAPER_BACKEND: str = os.environ.get("SCRAPER_BACKEND", "auto")
    # Playwright headless mode
    PLAYWRIGHT_HEADLESS: bool = os.environ.get("PLAYWRIGHT_HEADLESS", "1") == "1"

# Default output directory (can be overridden via env or Config)
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def url_to_filename(url: str, suffix: str = "") -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{h}{suffix}"


def save_bytes(output_dir: str, filename: str, data: bytes):
    ensure_dir(output_dir)
    Path(output_dir, filename).write_bytes(data)


def save_text(output_dir: str, filename: str, text: str):
    ensure_dir(output_dir)
    Path(output_dir, filename).write_text(text, encoding='utf-8')


def save_json(output_dir: str, filename: str, obj):
    ensure_dir(output_dir)
    Path(output_dir, filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def save_pickle(output_dir: str, filename: str, obj):
    ensure_dir(output_dir)
    with open(Path(output_dir, filename), 'wb') as f:
        pickle.dump(obj, f)


def save_chat_history(output_dir: str, chat_history):
    try:
        filename = f"chat_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        save_json(os.path.join(output_dir, 'chats'), filename, chat_history)
    except Exception as e:
        logger.warning(f"Could not save chat history: {e}")


class WebScraper:
    """
    Streamlit Cloud Compatible Scraper
    Supports requests, curl_cffi, or Playwright to bypass Cloudflare when needed.
    """
    def __init__(self, config: Optional[Config] = None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.config = config or Config()
        self.backend = (self.config.SCRAPER_BACKEND or 'auto').lower()
        self.playwright_headless = getattr(self.config, 'PLAYWRIGHT_HEADLESS', True)

    def _looks_like_cloudflare(self, text: str, status: int) -> bool:
        if status in (403, 429, 503):
            return True
        if not text:
            return True
        low = text.lower()
        checks = [
            'cloudflare',
            'checking your browser',
            'attention required',
            'cf-chl-bypass',
            'hit 1 sec'
        ]
        return any(c in low for c in checks)

    def _fetch_requests(self, url: str, timeout: int = 15):
        try:
            resp = requests.get(url, headers=self.headers, timeout=timeout)
            return {'status_code': getattr(resp, 'status_code', 200), 'content': getattr(resp, 'content', b''), 'text': getattr(resp, 'text', '')}
        except Exception as e:
            raise

    def _fetch_curl_cffi(self, url: str, timeout: int = 15):
        if not CURL_CFFI_AVAILABLE:
            raise ImportError('curl_cffi not available')
        resp = cc_requests.get(url, headers=self.headers, timeout=timeout)
        return {'status_code': getattr(resp, 'status_code', 200), 'content': getattr(resp, 'content', b''), 'text': getattr(resp, 'text', '')}

    def _fetch_playwright(self, url: str, timeout: int = 30):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError('Playwright not available')
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.playwright_headless)
            page = browser.new_page(user_agent=self.headers.get('User-Agent'))
            try:
                page.set_default_navigation_timeout(timeout * 1000)
                page.goto(url, wait_until='networkidle')
                content = page.content()
                # Playwright does not expose status easily for page content; assume 200 on success
                return {'status_code': 200, 'content': content.encode('utf-8'), 'text': content}
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a given URL using requests
        """
        try:
            logger.info(f"Scraping URL: {url} using backend={self.backend}")

            last_exc = None
            resp = None

            # Helper to try a fetcher and set resp
            def try_fetch(fetch_fn, *a, **kw):
                nonlocal last_exc, resp
                try:
                    resp = fetch_fn(*a, **kw)
                    return True
                except Exception as e:
                    last_exc = e
                    logger.debug(f"Fetcher {fetch_fn.__name__} failed: {e}")
                    return False

            # Decide order
            tried = []
            if self.backend == 'requests':
                try_fetch(self._fetch_requests, url)
                tried = ['requests']
            elif self.backend == 'curl_cffi':
                try_fetch(self._fetch_curl_cffi, url)
                tried = ['curl_cffi']
            elif self.backend == 'playwright':
                try_fetch(self._fetch_playwright, url)
                tried = ['playwright']
            else:  # auto
                # Try requests first
                if try_fetch(self._fetch_requests, url):
                    tried.append('requests')
                    if self._looks_like_cloudflare(resp.get('text', ''), resp.get('status_code', 200)):
                        logger.info('Detected possible Cloudflare/JS protection; trying curl_cffi')
                        if CURL_CFFI_AVAILABLE and try_fetch(self._fetch_curl_cffi, url):
                            tried.append('curl_cffi')
                        elif PLAYWRIGHT_AVAILABLE and try_fetch(self._fetch_playwright, url):
                            tried.append('playwright')
                else:
                    # try curl_cffi then playwright
                    if CURL_CFFI_AVAILABLE and try_fetch(self._fetch_curl_cffi, url):
                        tried.append('curl_cffi')
                    elif PLAYWRIGHT_AVAILABLE and try_fetch(self._fetch_playwright, url):
                        tried.append('playwright')

            if not resp:
                raise last_exc or RuntimeError('All fetchers failed')

            content_bytes = resp.get('content', b'') if resp.get('content', None) is not None else (resp.get('text', '') or '').encode('utf-8')
            soup = BeautifulSoup(content_bytes, 'html.parser')

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

            logger.info(f"Successfully scraped {len(text)} characters from {url} (via {','.join(tried)})")

            # Save artifacts: raw HTML, cleaned text, and metadata
            try:
                html_filename = url_to_filename(url, '.html')
                save_bytes(os.path.join(DEFAULT_OUTPUT_DIR, 'html'), html_filename, content_bytes)
                text_filename = url_to_filename(url, '.txt')
                save_text(os.path.join(DEFAULT_OUTPUT_DIR, 'text'), text_filename, text)
                metadata_filename = url_to_filename(url, '.json')
                meta = {'url': url, 'title': title_text, 'chars': len(text), 'saved_at': datetime.utcnow().isoformat(), 'backend_tried': tried}
                save_json(os.path.join(DEFAULT_OUTPUT_DIR, 'metadata'), metadata_filename, meta)
            except Exception as e:
                logger.warning(f"Could not save scraped files: {e}")

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

        # Save chunks to disk as JSON
        try:
            chunks_list = []
            for i, c in enumerate(chunks):
                chunks_list.append({
                    'index': i,
                    'content': c.page_content,
                    'metadata': c.metadata
                })
            chunks_filename = url_to_filename(scraped_data['url'], '_chunks.json')
            save_json(os.path.join(DEFAULT_OUTPUT_DIR, 'chunks'), chunks_filename, chunks_list)
        except Exception as e:
            logger.warning(f"Could not save chunks: {e}")

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

        # Save vector store and embeddings
        try:
            store_name = url_to_filename('vectorstore', '')
            store_dir = Path(DEFAULT_OUTPUT_DIR) / 'vector_store' / store_name
            ensure_dir(store_dir.parent)
            # Try to use FAISS save_local if available
            if hasattr(self.vector_store, "save_local"):
                self.vector_store.save_local(str(store_dir))
                logger.info(f"Saved vector store to {store_dir}")
            else:
                # Fallback to pickle
                save_pickle(str(Path(DEFAULT_OUTPUT_DIR) / 'vector_store'), f"{store_name}.pkl", self.vector_store)
                logger.info(f"Pickled vector store to {Path(DEFAULT_OUTPUT_DIR) / 'vector_store' / (store_name + '.pkl')}")
        except Exception as e:
            logger.warning(f"Could not save vector store: {e}")

        # Save embeddings separately
        try:
            texts = [d.page_content for d in documents]
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_arr = np.array(embeddings, dtype=np.float32)
            embed_filename = url_to_filename('embeddings', '.npy')
            ensure_dir(Path(DEFAULT_OUTPUT_DIR) / 'embeddings')
            np.save(Path(DEFAULT_OUTPUT_DIR) / 'embeddings' / embed_filename, embeddings_arr)
            # Save metadata mapping
            meta = [{'index': i, 'metadata': d.metadata} for i, d in enumerate(documents)]
            save_json(os.path.join(DEFAULT_OUTPUT_DIR, 'embeddings'), url_to_filename('embeddings_meta', '.json'), meta)
            logger.info(f"Saved embeddings to {Path(DEFAULT_OUTPUT_DIR) / 'embeddings' / embed_filename}")
        except Exception as e:
            logger.warning(f"Could not save embeddings: {e}")

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

                    # Save updated chat history to disk
                    try:
                        save_chat_history(st.session_state.config.OUTPUT_DIR, st.session_state.chat_history)
                    except Exception as e:
                        logger.warning(f"Could not auto-save chat history: {e}")

if __name__ == "__main__":
    main()