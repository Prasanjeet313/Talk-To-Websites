# Complete Installation Guide - FINAL FIX

## ⚠️ IMPORTANT: Follow These Steps Exactly

The LangChain package structure has changed significantly. Here's the complete fix:

---

## ✅ SOLUTION: Complete Clean Install

### Step 1: Install All Required Packages

Run this command to install ALL packages at once:

```bash
pip install streamlit langchain langchain-core langchain-community langchain-text-splitters langchain-groq faiss-cpu sentence-transformers undetected-chromedriver selenium beautifulsoup4 requests lxml torch transformers huggingface-hub typing-extensions python-dotenv
```

### OR use the requirements file:

```bash
pip install -r requirements_simple.txt
```

---

## 🔧 If You Get Import Errors

### Install Packages One by One:

```bash
# Core
pip install streamlit
pip install python-dotenv

# Web scraping
pip install requests
pip install beautifulsoup4
pip install lxml
pip install undetected-chromedriver
pip install selenium

# LangChain (in this exact order!)
pip install langchain-core
pip install langchain-community
pip install langchain-text-splitters
pip install langchain
pip install langchain-groq

# ML/AI
pip install faiss-cpu
pip install sentence-transformers
pip install torch
pip install transformers
pip install huggingface-hub

# Utils
pip install typing-extensions
```

---

## 🎯 Test Your Installation

After installation, run this test:

```bash
python -c "
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
print('✅ ALL IMPORTS SUCCESSFUL!')
"
```

If you see "✅ ALL IMPORTS SUCCESSFUL!", you're good to go!

---

## 🚀 Run the App

```bash
streamlit run webpage_rag_chatbot_simple.py
```

---

## 💡 BEST PRACTICE: Use Virtual Environment

To avoid conflicts with other projects:

```bash
# Create fresh virtual environment
python -m venv bankbot_env

# Activate it
# Windows:
bankbot_env\Scripts\activate
# Linux/Mac:
source bankbot_env/bin/activate

# Install all packages
pip install -r requirements_simple.txt

# Run app
streamlit run webpage_rag_chatbot_simple.py
```

---

## 📋 Package Versions That Work Together

```
streamlit >= 1.30.0
langchain >= 0.1.0
langchain-core >= 0.1.0
langchain-community >= 0.0.20
langchain-text-splitters >= 0.0.1
langchain-groq >= 0.1.0
faiss-cpu >= 1.7.4
sentence-transformers >= 2.2.0
undetected-chromedriver >= 3.5.0
selenium >= 4.15.0
```

---

## 🔍 Common Import Issues & Fixes

### Error: `No module named 'langchain.chains'`
**Fix:**
```bash
pip install langchain langchain-core
```

### Error: `No module named 'langchain_text_splitters'`
**Fix:**
```bash
pip install langchain-text-splitters
```

### Error: `No module named 'langchain_core'`
**Fix:**
```bash
pip install langchain-core
```

### Error: `No module named 'langchain_groq'`
**Fix:**
```bash
pip install langchain-groq
```

---

## 🎮 Quick Command (Copy-Paste This!)

```bash
pip install streamlit langchain langchain-core langchain-community langchain-text-splitters langchain-groq faiss-cpu sentence-transformers undetected-chromedriver selenium beautifulsoup4 requests lxml && streamlit run webpage_rag_chatbot_simple.py
```

This single command:
1. Installs all required packages
2. Runs the app

---

## ⚡ Ultra-Fast Fix (If Nothing Else Works)

```bash
# 1. Uninstall everything LangChain related
pip uninstall langchain langchain-core langchain-community langchain-text-splitters langchain-groq -y

# 2. Reinstall in correct order
pip install langchain-core
pip install langchain-community
pip install langchain-text-splitters
pip install langchain
pip install langchain-groq

# 3. Run app
streamlit run webpage_rag_chatbot_simple.py
```

---

## ✅ What Changed in the Code

The imports were updated to match the new LangChain structure:

### Old (Broken):
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
```

### New (Working):
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
```

---

## 📦 Updated Files

Download the latest versions:
1. ✅ `webpage_rag_chatbot_simple.py` - All imports fixed
2. ✅ `requirements_simple.txt` - All packages included

---

## 🎯 Final Checklist

- [ ] Install packages from requirements_simple.txt
- [ ] Run the Python test command above
- [ ] See "✅ ALL IMPORTS SUCCESSFUL!"
- [ ] Run: `streamlit run webpage_rag_chatbot_simple.py`
- [ ] App opens in browser
- [ ] No import errors

---

## 🆘 Still Not Working?

Try this nuclear option:

```bash
# Create completely fresh environment
python -m venv fresh_env
fresh_env\Scripts\activate  # Windows
# or
source fresh_env/bin/activate  # Linux/Mac

# Upgrade pip
pip install --upgrade pip

# Install from scratch
pip install -r requirements_simple.txt

# Run
streamlit run webpage_rag_chatbot_simple.py
```

---

**This should fix ALL import errors! 🎉**

Just copy-paste the commands and you're done!
