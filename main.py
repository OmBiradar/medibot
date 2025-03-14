import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List
import google.generativeai as genai

# Custom Gemini LLM Wrapper
class GeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash"  # Verify this model exists
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

# Setup
search_tool = DuckDuckGoSearchRun()
query = "How is hypertension treated?"
search_results = search_tool.run(query)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)  # Increased chunk size
docs = text_splitter.create_documents([search_results])

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embedding_model)

gemini_api_key = os.getenv("GEMINI_API_KEY")  # Use env variable
llm = GeminiLLM(api_key=gemini_api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Test
response = qa_chain.invoke({"query": query})
print("Answer:", response["result"])
print("Sources:", [doc.page_content for doc in response["source_documents"]])