from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from fastapi import FastAPI, File, UploadFile, Form

from langchain_openai import ChatOpenAI

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import BaseMessage
from addknowledgeBase import add_knowledgebase
from core.cong import key  
# Load API key from .env
load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    print("API_KEY is not set in the environment variables.")
else:
    print(key, "API_KEY is set in the environment variables.")

# Init FastAPI
app = FastAPI()

# Init OpenAI model
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")

# Input model
class PromptInput(BaseModel):
    user_input: str

# Endpoint with ChatPromptTemplate
@app.post("/chat")
async def chat_with_prompt_template(request: PromptInput):
    try:
        # Define a reusable prompt template
        system_template = "You are a helpful assistant who responds politely and accurately."
        user_template = "{user_input}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
        ])

        # Format prompt with user input
        chat_messages: List[BaseMessage] = prompt.format_messages(user_input=request.user_input)

        # Get model response
        response = llm(chat_messages)

        return {"response": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-knowledgebase")
async def knowledgebase(
    file: UploadFile = File(...),  # Required
    collection_name: str = Form(...)  # Required
):
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was uploaded."
        )
    if not collection_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Collection name is required."
        )
    
    try:
        result = await add_knowledgebase(file, collection_name)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the knowledgebase: {str(e)}")

@app.post("/rag-chat")
async def add(query: str = Form(...), temp: float = 0, collection_name: str = Form(...)):
    persist_directory = f'database/{collection_name}'

    try:
        if not os.path.exists(persist_directory) or not os.path.isdir(persist_directory):
            return {"status": "error", "message": f"Directory '{persist_directory}' does NOT exist."}

        print(f"✅ Persist directory {persist_directory} exists.")

        embeddings = OpenAIEmbeddings()

        db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        template = """
        Prompt:
        You are a multilingual expert assistant that generates detailed and well-structured answers using only data stored in ChromaDB. Always follow the format and rules below:
        Format:
        Heading: Clearly state the topic or subtopic.
        Paragraphs: Provide detailed explanations, insights, and relevant information under each heading.
        References: Always include references from ChromaDB with the correct format:
        (PDF Name).pdf – Page X
        Language Rules:
        If the user asks in English, respond entirely in English.
        Then provide references followed by ChromaDB.
        Responsibilities:
        -Always extract data from ChromaDB.
        -Always include all references from ChromaDB with the correct PDF name and page number.
        -Never use pre-trained knowledge unless it is also found in ChromaDB.
        -Do not show internal reference IDs like “ChromaDB Reference ID: 001”.
        -Always end the answer with a References section in this format:
        (PDF Name).pdf – Page X
        (PDF Name).pdf – Page Y
        -Think step-by-step before providing the final answer.
        -Always use headings, structured paragraphs, and clear formatting.
        -Each response should be highly detailed and exceed 3000 words when requested.

        <context>
            {context}
        </context>

        Question: {question}
        """
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        llm = ChatOpenAI(max_tokens=1000, temperature=temp, model="gpt-4o")

        total_docs = db._collection.count()
        print(f"Total documents in vector store: {total_docs}")
        fetch_k = min(total_docs, 1000)
        k = min(fetch_k, 20)

        qa_with_source = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': k, 'fetch_k': fetch_k}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )

        result = qa_with_source.invoke({"query": query})
        return {"response": result}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the RAG chat: {str(e)}"
        )