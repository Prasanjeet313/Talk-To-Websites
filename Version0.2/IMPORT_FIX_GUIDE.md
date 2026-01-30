# 🔧 Quick Fix - Import Error

## Problem
```
ModuleNotFoundError: No module named 'langchain.text_splitter'
```

## ✅ Solution

The issue is a missing package. Run this command:

```bash
pip install langchain-text-splitters
```

Or reinstall all dependencies:

```bash
pip install -r requirements_simple.txt
```

---

## Complete Installation Steps

### Step 1: Clean Install (Recommended)

```bash
# Uninstall old packages
pip uninstall langchain langchain-community langchain-groq -y

# Install fresh from requirements
pip install -r requirements_simple.txt
```

### Step 2: Verify Installation

```bash
python -c "from langchain_text_splitters import RecursiveCharacterTextSplitter; print('✅ Import successful')"
```

### Step 3: Run the App

```bash
streamlit run webpage_rag_chatbot_simple.py
```

---

## If Still Having Issues

### Option 1: Install Packages Individually

```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-groq
pip install faiss-cpu
pip install sentence-transformers
pip install undetected-chromedriver
pip install selenium
pip install beautifulsoup4
pip install requests
pip install lxml
```

### Option 2: Use Virtual Environment (Best Practice)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements_simple.txt

# Run app
streamlit run webpage_rag_chatbot_simple.py
```

---

## Package Versions

Make sure you have compatible versions:

```
langchain >= 0.1.0
langchain-community >= 0.0.20
langchain-text-splitters >= 0.0.1
langchain-groq >= 0.1.0
```

---

## Quick Test

After installation, test if imports work:

```python
# Test script
python -c "
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
print('✅ All imports successful!')
"
```

If this works, the app should run fine!

---

## Common Issues

### Issue: "No module named 'langchain_text_splitters'"
**Fix:**
```bash
pip install langchain-text-splitters
```

### Issue: "No module named 'langchain_groq'"
**Fix:**
```bash
pip install langchain-groq
```

### Issue: Multiple version conflicts
**Fix:** Use a fresh virtual environment
```bash
python -m venv venv_fresh
venv_fresh\Scripts\activate  # Windows
pip install -r requirements_simple.txt
```

---

## Updated Files

I've updated these files to fix the import issue:

1. ✅ **webpage_rag_chatbot_simple.py** - Fixed imports
2. ✅ **requirements_simple.txt** - Added missing package

Download the latest versions and reinstall!

---

**Quick Command:**
```bash
pip install -r requirements_simple.txt && streamlit run webpage_rag_chatbot_simple.py
```

Should work now! 🎉
