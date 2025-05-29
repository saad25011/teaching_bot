import os
import shutil
from fastapi import UploadFile
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from core.cong import key  # Your OpenAI API key

async def add_knowledgebase(file: UploadFile, collection_name: str):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load PDF content using Unstructured
        loader = UnstructuredPDFLoader(temp_file_path)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s) from the PDF.")

        # Remove the temporary file
        os.remove(temp_file_path)

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=key)

        # Semantic Chunking
        text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
        docs_chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(docs_chunks)} semantic chunks.")

        # Define vector store path
        persist_directory = f'database/{collection_name}'
        os.makedirs(persist_directory, exist_ok=True)

        # Save chunks to Chroma
        db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        db.add_documents(docs_chunks)
        db.persist()

        return {
            "message": "✅ File processed and chunks uploaded.",
            "collection_name": collection_name,
            "chunks": len(docs_chunks)
        }

    except Exception as e:
        return {
            "error": f"❌ Failed to process knowledgebase: {str(e)}"
        }
