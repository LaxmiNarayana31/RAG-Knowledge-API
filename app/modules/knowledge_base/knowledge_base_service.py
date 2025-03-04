from pathlib import Path
from fastapi import UploadFile
from app.helper.llm_helper import GoogleGeminiEmbeddings
from typing import List
import os
from io import BytesIO
from app.helper.ai_helper import AIHelper
from uuid import uuid4
from langchain_community.vectorstores import FAISS
import shutil
from app.dto.knowledge_schema import AskQuestion, deleteKnowledgeBase
from app.utils.exception_handler import handle_exception


class KnowledgeBaseService:
    # Create knowledge base 
    def createKnowledgeBase(documents: List[UploadFile]):
        try:
            upload_dir = "uploads"
            document_buffers = []
            document_uuid = str(uuid4())

            for document in documents:
                if not document:
                    return "No file uploaded"

                document_content = document.file.read()
                
                unique_filename = f"{document.filename}" 
                file_path = os.path.join(upload_dir, unique_filename)

                # Save the uploaded file to disk
                with open(file_path, "wb") as buffer:
                    buffer.write(document_content)

                print(f"Uploaded file saved at: {file_path}")
                
                document_buffer = BytesIO(document_content)
                document_buffer.name = unique_filename 
                document_buffers.append(document_buffer)

            if document_buffers:
                AIHelper.build_vectorstore(document_files = document_buffers, document_uuid = document_uuid)

            # Remove the uploaded files from the server
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            for document_buffer in document_buffers:  
                document_buffer.close()

            response = {"document_uuid": document_uuid}
            return response
        except Exception as e:
            return handle_exception(e)
        

    # Ask question on knowledge base    
    def askQuestion(question: AskQuestion):
        try:
            llm_response =  AIHelper.get_llm_response(user_question = question.question)
            response = {"llm_response": llm_response}
            return response
        except Exception as e:
            return handle_exception(e)


    # Retrieve all stored documents and their metadata from knowledge base 
    def getAllKnowledgeBasesList():
        try:
            embeddings = GoogleGeminiEmbeddings()
            vectorstore_path = Path(AIHelper.vectorstore_folder)

            if not vectorstore_path.exists():
                return 1

            vectorstore = FAISS.load_local(folder_path = vectorstore_path, embeddings = embeddings, allow_dangerous_deserialization = True)

            docs = list(vectorstore.docstore._dict.values())

            unique_documents = {
                (doc.metadata.get("document_name", "Unknown"), doc.metadata.get("document_uuid", "Unknown")) 
                for doc in docs
            }

            stored_documents = []
            for document_name, document_uuid in unique_documents:
                stored_documents.append({"document_name": document_name, "document_uuid": document_uuid})

            response = {"stored_documents": stored_documents}
            return response
        except Exception as e:
            return handle_exception(e)
            
            
    # Delete document knowledge base
    def deleteKnowledgeBase(document_uuid: deleteKnowledgeBase):
        try:
            document_uuid = document_uuid.doc_uuid
            vectorstore_path = Path(AIHelper.vectorstore_folder)
            if vectorstore_path.exists() == False:
                return 1    
            embeddings = GoogleGeminiEmbeddings() 
            vectorstore = FAISS.load_local(folder_path = vectorstore_path, embeddings = embeddings, allow_dangerous_deserialization = True)

            if document_uuid:
                vector_ids_to_delete = []
                for key, doc in vectorstore.docstore._dict.items():
                    if doc.metadata.get("document_uuid") == document_uuid:
                        vector_ids_to_delete.append(key)
                # vector_ids_to_delete = [key for key, doc in vectorstore.docstore._dict.items() if doc.metadata.get("document_uuid") == document_uuid]
                if not vector_ids_to_delete:
                    return 2 
                
                vectorstore.delete(vector_ids_to_delete)
                vectorstore.save_local(AIHelper.vectorstore_folder)
                return 3
            
            shutil.rmtree(AIHelper.vectorstore_folder) 
            return None
        except Exception as e:
            return handle_exception(e)
