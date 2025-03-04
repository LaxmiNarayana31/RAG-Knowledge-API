# app/helper/llm_helper.py

import os 
from dotenv import load_dotenv
import google.generativeai as genai
# from langchain_ollama.chat_models import ChatOllama
from langchain_community.chat_models.ollama import ChatOllama 
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

load_dotenv(verbose = True)

# Custom embedding class using google gemini 
class GoogleGeminiEmbeddings:
    def __call__(self, text: str):
        return self.embed_query(text)
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document" 
            )
            embeddings.append(result['embedding'])  
        return embeddings

    def embed_query(self, query):
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_document"
        )
        return result['embedding']


# Custom LLM class for generate response from different LLMs
class LlmHelper():
    def googleGeminiLlm():
        gemini_model = os.getenv("MODEL_NAME")
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model = gemini_model, 
            google_api_key = gemini_api_key,
            temperature = 0.3, top_p = 0.9

        )
        return llm

    def metaLlamaLlm():
        llama_model = os.getenv("LLAMA_MODEL")
        llama_host_url = os.getenv("LLAMA_BASE_URL")
        llm = ChatOllama(
            model = llama_model, 
            base_url = llama_host_url,
            temperature = 0.5, top_k = 0.60, top_p = 0.95
        )
        return llm
    
    def deepSeekLlm():
        deepseek_model = os.getenv("DEEPSEEK_MODEL")
        llama_host_url = os.getenv("LLAMA_BASE_URL")
        llm = ChatOllama(
            model = deepseek_model,
            base_url = llama_host_url,
            temperature = 0.5, top_k = 0.60, top_p = 0.95
        )
        return llm
    