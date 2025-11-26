"""
ingest_faiss.py
This script creates a FAISS vector database from your MVA_Data.md file
Run this ONCE before running the Streamlit app
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import os

print("🔄 Starting data ingestion process...")

# Step 1: Load the MVA data from markdown file (WITHOUT document loader)
print("📄 Loading MVA_Data.md...")
try:
    # Direct file reading - no document loader needed!
    with open("MVA_Data.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a Document object manually
    documents = [Document(page_content=content, metadata={"source": "MVA_Data.md"})]
    print(f"✅ Loaded {len(documents)} document(s)")
    print(f"   File size: {len(content)} characters")
except FileNotFoundError:
    print("❌ Error: MVA_Data.md not found!")
    print("   Make sure the file exists in the current directory.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# Step 2: Split documents into smaller chunks
print("✂️  Splitting text into chunks...")
text_splitter = CharacterTextSplitter(
    separator='-->',  # Split at MVA section markers
    is_separator_regex=False,
    chunk_size=500,
    chunk_overlap=50  # Add overlap to maintain context
)
texts = text_splitter.split_documents(documents)
print(f"✅ Created {len(texts)} text chunks")

# Step 3: Load embedding model
print("🤖 Loading embedding model (BAAI/bge-base-en)...")
print("   ⏳ This may take a few minutes on first run (downloading 440MB model)...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embedding model loaded")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    print("💡 Try: pip install sentence-transformers torch")
    exit(1)

# Step 4: Create FAISS vector store
print("💾 Creating FAISS vector database...")
print("   ⏳ This may take 1-2 minutes...")
try:
    vectorstore = FAISS.from_documents(
        documents=texts,
        embedding=embeddings
    )
    print("✅ FAISS vector database created")
except Exception as e:
    print(f"❌ Error creating FAISS index: {e}")
    print("💡 Try: pip install faiss-cpu")
    exit(1)

# Step 5: Save to disk
print("💿 Saving to disk...")
try:
    vectorstore.save_local("faiss_mva_index")
    print("✅ FAISS index saved to 'faiss_mva_index' folder")
except Exception as e:
    print(f"❌ Error saving index: {e}")
    exit(1)

print("\n" + "="*50)
print("🎉 DATA INGESTION COMPLETE!")
print("="*50)
print(f"📊 Statistics:")
print(f"   - Documents loaded: {len(documents)}")
print(f"   - Text chunks created: {len(texts)}")
print(f"   - Embedding model: BAAI/bge-base-en (768 dimensions)")
print(f"   - Index location: faiss_mva_index/")
print("\n📌 Next step:")
print("   streamlit run streamlit_app.py")
print("="*50)