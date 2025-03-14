import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from typing import Optional, List
import google.generativeai as genai

# Step 1: Custom Gemini LLM Wrapper
class GeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash"  # Verify this model exists as of March 2025
    api_key: str

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        genai.configure(api_key=api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "api_key": self.api_key}

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Step 2: Read Medical Records File and Setup Initial Data
file_path = "medical_records.txt"
try:
    with open(file_path, "r") as file:
        medical_records = file.read()
except FileNotFoundError:
    medical_records = "No medical records found. Please ensure 'medical_records.txt' exists."
    print(medical_records)

# Split the medical records into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.create_documents([medical_records])

# Create embeddings and vector store (global variable to update later)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embedding_model)

# Step 3: Initialize Gemini LLM
gemini_api_key = os.getenv("GEMINI_API_KEY", "your_api_key_here")  # Replace with your key or set env variable
llm = GeminiLLM(api_key=gemini_api_key)

# Step 4: Add Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Step 5: Function to Generate Patient-Friendly Summary
def generate_summary(llm, medical_records):
    prompt = f"""
    You are a helpful medical assistant. Below is a medical report from a patient's records. Please read it and provide a summary in simple, patient-friendly language that explains their case clearly. Avoid using complex medical jargon unless necessary, and if you do, explain it simply.

    Medical Report:
    {medical_records}

    Summary:
    """
    return llm._call(prompt)

# Step 6: Function to Check if Web Search is Needed
def needs_web_search(question: str, vector_store, llm) -> bool:
    # Retrieve top 3 relevant documents from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Prompt the LLM with the retrieved context
    prompt = f"""
    I have a vector store with information from a patient's medical records and possibly additional web searches.
    The user has asked: '{question}'.
    Here’s the most relevant information I currently have from the vector store:
    
    {context}
    
    Based on this, do you think the current information is sufficient to answer the question accurately,
    or should I perform a new web search to gather more relevant data? Answer with 'yes' (sufficient) or 'no' (needs search).
    """
    response = llm._call(prompt).strip().lower()
    return response == "no"

# Step 7: Function to Update Vector Store with New Search
search_tool = DuckDuckGoSearchRun()
def update_vector_store(query: str, vector_store, search_tool, text_splitter, embedding_model):
    new_search_results = search_tool.run(query)
    new_docs = text_splitter.create_documents([new_search_results])
    vector_store.add_documents(new_docs)
    return vector_store

# Step 8: Create Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# Step 9: Test the System with a Conversation Loop
def run_conversation():
    global vector_store
    print("Start chatting! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Check if a new web search is needed based on vector store content
        if needs_web_search(user_input, vector_store, llm):
            print(f"Bot: I need more information to answer that. Searching the web for '{user_input}'...")
            vector_store = update_vector_store(user_input, vector_store, search_tool, text_splitter, embedding_model)
            qa_chain.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Run the chain with the user's input
        response = qa_chain.invoke({"question": user_input})
        
        # Print the answer and (optionally) sources
        print("Bot:", response["answer"])
        # Uncomment the line below to see sources
        # print("Sources:", [doc.page_content for doc in response["source_documents"]])

if __name__ == "__main__":
    # Initial summary of medical records
    initial_query = "Summarize my medical records for me."
    summary = generate_summary(llm, medical_records)
    print("Summary for Patient:", summary)
    
    # Populate memory with the initial summary
    response = qa_chain.invoke({"question": initial_query})
    print("Bot Response:", response["answer"])
    # Uncomment the line below to see sources
    # print("Sources:", [doc.page_content for doc in response["source_documents"]])
    
    # Start interactive conversation
    run_conversation()