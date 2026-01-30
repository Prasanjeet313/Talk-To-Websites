"""
Webpage RAG Chatbot - A Streamlit application for querying webpage content
Uses LangChain, FAISS, and Groq LLM (Simplified version without LangGraph)
"""

import os
import streamlit as st
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import logging

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for the application"""
    GROQ_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL: str = "mixtral-8x7b-32768"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048


class WebScraper:
    """Handles web scraping operations with Cloudflare bypass using undetected-chromedriver"""
    
    def __init__(self, headless: bool = False, use_selenium: bool = True):
        """
        Initialize web scraper
        
        Args:
            headless: Run browser in headless mode (may trigger Cloudflare)
            use_selenium: Use Selenium for JavaScript-heavy sites (slower but more reliable)
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.headless = headless
        self.use_selenium = use_selenium
        self.driver = None
        
        if use_selenium:
            self._init_driver()
    
    def _init_driver(self):
        """Initialize undetected Chrome WebDriver"""
        try:
            import undetected_chromedriver as uc
            
            logger.info("Initializing Chrome WebDriver...")
            options = uc.ChromeOptions()
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            if self.headless:
                options.add_argument('--headless')
                logger.warning("Running in headless mode (may trigger Cloudflare)")
            
            # Create driver
            try:
                self.driver = uc.Chrome(options=options, version_main=131)
            except:
                # Fallback without version_main if it causes issues
                self.driver = uc.Chrome(options=options)
            
            logger.info("Chrome WebDriver initialized successfully")
        except ImportError:
            logger.warning("undetected-chromedriver not installed. Install with: pip install undetected-chromedriver")
            logger.info("Falling back to requests-based scraping")
            self.use_selenium = False
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            logger.info("Falling back to requests-based scraping")
            self.use_selenium = False
    
    def scrape_url(self, url: str, wait_time: int = 15) -> Dict[str, Any]:
        """
        Scrape content from a given URL with Cloudflare bypass
        
        Args:
            url: The URL to scrape
            wait_time: Time to wait for page load and Cloudflare bypass (seconds)
            
        Returns:
            Dictionary containing title, text content, and metadata
        """
        if self.use_selenium and self.driver:
            return self._scrape_with_selenium(url, wait_time)
        else:
            return self._scrape_with_requests(url)
    
    def _scrape_with_selenium(self, url: str, wait_time: int = 15) -> Dict[str, Any]:
        """Scrape using Selenium (bypasses Cloudflare)"""
        import time
        
        try:
            logger.info(f"Scraping URL with Selenium: {url}")
            
            # Navigate to URL
            self.driver.get(url)
            logger.info(f"Waiting {wait_time} seconds for page to load and bypass Cloudflare...")
            
            # Wait for Cloudflare challenge
            time.sleep(wait_time)
            
            # Check if Cloudflare blocked us
            title_lower = self.driver.title.lower()
            if "just a moment" in title_lower or "checking" in title_lower:
                logger.warning("Cloudflare challenge still active, waiting longer...")
                time.sleep(10)
            elif "403" in title_lower or "forbidden" in title_lower:
                logger.error("Access forbidden (403)")
                return {
                    'url': url,
                    'title': '',
                    'content': '',
                    'success': False,
                    'error': 'Access forbidden (403)'
                }
            
            page_title = self.driver.title
            logger.info(f"Page Title: {page_title}")
            
            # Get page source
            page_source = self.driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            
            return {
                'url': url,
                'title': page_title,
                'content': text,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error scraping with Selenium {url}: {str(e)}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': str(e)
            }
    
    def _scrape_with_requests(self, url: str) -> Dict[str, Any]:
        """Scrape using requests (fallback, may not work with Cloudflare)"""
        try:
            logger.info(f"Scraping URL with requests: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Get main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
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
    
    def close(self):
        """Close the WebDriver if it's running"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed")
            except:
                pass


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
        """
        Convert scraped data into LangChain documents and chunk them
        
        Args:
            scraped_data: Dictionary containing scraped webpage data
            
        Returns:
            List of Document objects
        """
        if not scraped_data.get('success'):
            logger.warning("Cannot process unsuccessful scrape")
            return []
        
        # Create metadata
        metadata = {
            'source': scraped_data['url'],
            'title': scraped_data['title']
        }
        
        # Create document
        document = Document(
            page_content=scraped_data['content'],
            metadata=metadata
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([document])
        logger.info(f"Created {len(chunks)} document chunks")
        
        return chunks


class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self, embedding_model: str, use_gpu: bool = False):
        logger.info(f"Initializing embeddings with model: {embedding_model}")
        
        # Try GPU first if requested, fallback to CPU
        device = 'cuda' if use_gpu else 'cpu'
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Embeddings initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize on {device}, falling back to CPU: {str(e)}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.vector_store: Optional[FAISS] = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        logger.info("Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        logger.info("Vector store created successfully")
        return self.vector_store
    
    def get_retriever(self, k: int = 4):
        """Get retriever from vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


class RAGChatbot:
    """Main RAG chatbot using LangChain and Groq"""
    
    def __init__(self, config: Config):
        self.config = config
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
        """Create conversational retrieval chain"""
        
        # Custom prompt template
        template = """You are a helpful AI assistant specialized in answering questions about webpage content.
Use the following pieces of context from the webpage to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always provide detailed and accurate answers based on the webpage content.

Context from the webpage:
{context}

Chat History:
{chat_history}

Question: {question}

Helpful Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            verbose=False
        )
        
        return self.qa_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Please load a webpage first.")
        
        try:
            logger.info(f"Processing question: {question}")
            result = self.qa_chain({"question": question})
            
            return {
                'answer': result['answer'],
                'source_documents': result.get('source_documents', []),
                'success': True
            }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'answer': f"Error: {str(e)}",
                'source_documents': [],
                'success': False
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()


class WebpageRAGPipeline:
    """Complete pipeline for webpage RAG processing"""
    
    def __init__(self, config: Config, use_gpu: bool = False, use_selenium: bool = True, headless: bool = False):
        self.config = config
        self.scraper = WebScraper(headless=headless, use_selenium=use_selenium)
        self.processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_manager = VectorStoreManager(config.EMBEDDING_MODEL, use_gpu=use_gpu)
        self.chatbot = RAGChatbot(config)
    
    def process_url(self, url: str, wait_time: int = 15) -> Dict[str, Any]:
        """
        Process a URL through the complete pipeline
        
        Args:
            url: URL to process
            wait_time: Time to wait for page load (for Selenium mode)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Step 1: Scrape
            logger.info("Step 1/3: Scraping webpage...")
            scraped_data = self.scraper.scrape_url(url, wait_time=wait_time)
            
            if not scraped_data['success']:
                return {
                    'success': False,
                    'error': scraped_data.get('error', 'Scraping failed'),
                    'stage': 'scraping'
                }
            
            # Step 2: Process and chunk
            logger.info("Step 2/3: Processing and chunking documents...")
            documents = self.processor.process_scraped_data(scraped_data)
            
            if not documents:
                return {
                    'success': False,
                    'error': 'No documents created from scraped content',
                    'stage': 'processing'
                }
            
            # Step 3: Create vector store
            logger.info("Step 3/3: Creating vector store...")
            vector_store = self.vector_manager.create_vector_store(documents)
            
            # Step 4: Initialize chatbot
            retriever = self.vector_manager.get_retriever()
            self.chatbot.create_qa_chain(retriever)
            
            logger.info("Pipeline completed successfully!")
            
            return {
                'success': True,
                'title': scraped_data['title'],
                'num_chunks': len(documents),
                'content_length': len(scraped_data['content']),
                'url': url
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'pipeline'
            }
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question to the chatbot"""
        return self.chatbot.ask(question)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.chatbot.clear_memory()
    
    def close(self):
        """Close resources (browser, etc.)"""
        self.scraper.close()


# Streamlit UI
def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Webpage RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Webpage RAG Chatbot")
    st.markdown("### Chat with any webpage using AI")
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'webpage_loaded' not in st.session_state:
        st.session_state.webpage_loaded = False
    
    if 'current_url' not in st.session_state:
        st.session_state.current_url = ""
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Scraping options
        st.subheader("Scraping Options")
        use_selenium = st.checkbox(
            "Use Selenium (Cloudflare bypass)", 
            value=True, 
            help="Use undetected-chromedriver to bypass Cloudflare protection. Slower but more reliable."
        )
        
        headless = st.checkbox(
            "Headless mode", 
            value=False, 
            help="Run browser without GUI (may trigger Cloudflare detection)"
        )
        
        wait_time = st.slider(
            "Page load wait time (seconds)",
            min_value=5,
            max_value=30,
            value=15,
            help="Time to wait for page load and Cloudflare bypass"
        )
        
        # GPU option
        use_gpu = st.checkbox("Use GPU (if available)", value=False, help="Enable CUDA acceleration")
        
        st.divider()
        
        # URL Input
        url = st.text_input(
            "Enter Webpage URL",
            value="https://www.emiratesnbd.com/en/cards/credit-cards",
            help="Enter the URL of the webpage you want to chat with"
        )
        
        # Load button
        if st.button("🔄 Load Webpage", type="primary"):
            if url:
                with st.spinner("Processing webpage... This may take a moment."):
                    try:
                        # Close previous pipeline if exists
                        if st.session_state.pipeline:
                            try:
                                st.session_state.pipeline.close()
                            except:
                                pass
                        
                        # Initialize pipeline
                        pipeline = WebpageRAGPipeline(
                            st.session_state.config, 
                            use_gpu=use_gpu,
                            use_selenium=use_selenium,
                            headless=headless
                        )
                        
                        # Process URL
                        result = pipeline.process_url(url, wait_time=wait_time)
                        
                        if result['success']:
                            # Save to session state
                            st.session_state.pipeline = pipeline
                            st.session_state.webpage_loaded = True
                            st.session_state.current_url = url
                            st.session_state.chat_history = []
                            
                            st.success("✅ Webpage loaded successfully!")
                            st.info(f"**Title:** {result['title']}")
                            st.info(f"**Chunks created:** {result['num_chunks']}")
                            st.info(f"**Content length:** {result['content_length']:,} characters")
                        else:
                            error_msg = result.get('error', 'Unknown error occurred')
                            stage = result.get('stage', 'unknown')
                            st.error(f"❌ Error at {stage} stage: {error_msg}")
                            st.session_state.webpage_loaded = False
                            
                            # Close failed pipeline
                            try:
                                pipeline.close()
                            except:
                                pass
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state.webpage_loaded = False
            else:
                st.warning("⚠️ Please enter a URL")
        
        # Display current status
        st.divider()
        if st.session_state.webpage_loaded:
            st.success("✅ Webpage Ready")
            st.caption(f"URL: {st.session_state.current_url[:50]}...")
        else:
            st.info("ℹ️ No webpage loaded")
        
        # Clear chat button
        if st.session_state.webpage_loaded:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                if st.session_state.pipeline:
                    st.session_state.pipeline.clear_memory()
                st.rerun()
        
        # Model info
        st.divider()
        st.caption(f"**LLM Model:** {st.session_state.config.GROQ_MODEL}")
        st.caption(f"**Embedding Model:** {st.session_state.config.EMBEDDING_MODEL}")
        st.caption(f"**Chunk Size:** {st.session_state.config.CHUNK_SIZE}")
    
    # Main chat interface
    if st.session_state.webpage_loaded:
        st.markdown("---")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(message["sources"][:3], 1):
                            st.caption(f"**Source {i}:**")
                            st.text(doc.page_content[:300] + "...")
                            st.divider()
        
        # Chat input
        if question := st.chat_input("Ask a question about the webpage..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.pipeline.ask(question)
                    
                    if response['success']:
                        answer = response['answer']
                        st.markdown(answer)
                        
                        # Display sources
                        if response['source_documents']:
                            with st.expander("📚 View Sources"):
                                for i, doc in enumerate(response['source_documents'][:3], 1):
                                    st.caption(f"**Source {i}:**")
                                    st.text(doc.page_content[:300] + "...")
                                    st.divider()
                        
                        # Add to chat history with sources
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": response['source_documents']
                        })
                    else:
                        answer = response['answer']
                        st.error(answer)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": answer
                        })
    
    else:
        # Welcome message
        st.info("👈 Please load a webpage from the sidebar to start chatting!")
        
        st.markdown("""
        ### How to use:
        1. Enter a webpage URL in the sidebar
        2. Click "Load Webpage" to process the content
        3. Start asking questions about the webpage!
        
        ### Example Questions:
        - What credit cards are available?
        - What are the benefits of the Platinum card?
        - What is the annual fee?
        - How can I apply for a card?
        
        ### Tips:
        - Use specific questions for better answers
        - The chatbot remembers conversation context
        - Check sources to verify information
        """)


if __name__ == "__main__":
    main()
