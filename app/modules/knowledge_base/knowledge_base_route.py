from fastapi import APIRouter, File, UploadFile
from app.dto.response_schema import ResponseSchema
from app.dto.knowledge_schema import AskQuestion, deleteKnowledgeBase
from typing import List
from app.modules.knowledge_base.knowledge_base_service import KnowledgeBaseService
from config.response_msg import msg 

router = APIRouter(prefix = "/api/v1/knowledge-base", tags = ["Knowledge Base"])

@router.post("/add", summary = "Create a knowledge base")
def createKnowledgeBase(documents: List[UploadFile] = File(...)):
    response = KnowledgeBaseService.createKnowledgeBase(documents = documents)

    if response is not None and type(response) == dict:
        return ResponseSchema(status = True, response = msg["knowledge_base_created"], data = response)
    else:
        return ResponseSchema(status = False, response = msg["knowledge_base_creation_failed"], data = None)


@router.post("/ask_question", summary = "Ask a question")
def askQuestion(question: AskQuestion):
    response = KnowledgeBaseService.askQuestion(question = question)

    if response is not None and type(response) != int:
        return ResponseSchema(status = True, response = msg["questions_answered"], data = response)
    else:
        return ResponseSchema(status = False, response = msg["questions_not_answered"], data = None)


@router.get("/list", summary = "Get all knowledge base documents")
def getAllKnowledgeBasesList():
    response = KnowledgeBaseService.getAllKnowledgeBasesList()

    if response is not None and type(response) != int:
        return ResponseSchema(status = True, response = msg["knowledge_base_list_found"], data = response)
    elif response == 1 and type(response) == int:
        return ResponseSchema(status = False, response = msg["knowledge_base_not_found"], data = None)
    else:
        return ResponseSchema(status = False, response = msg["knowledge_base_list_not_found"], data = None)
    

@router.delete("/delete", summary = "Delete a knowledge base")
def deleteKnowledgeBase(document_uuid: deleteKnowledgeBase):
    response = KnowledgeBaseService.deleteKnowledgeBase(document_uuid = document_uuid)

    if response is None and type(response) != int:
        return ResponseSchema(status = True, response = msg["all_knowledge_bases_deleted"], data = None)
    elif response == 1 and type(response) == int:
        return ResponseSchema(status = False, response = msg["knowledge_base_not_found"], data = None)
    elif response == 2 and type(response) == int:
        return ResponseSchema(status = False, response = msg["document_uuid_not_found"], data = None)
    elif response == 3 and type(response) == int:
        return ResponseSchema(status = False, response = msg["knowledge_base_deleted"], data = None)
    else:
        return ResponseSchema(status = False, response = msg["knowledge_base_deletion_failed"], data = None)

