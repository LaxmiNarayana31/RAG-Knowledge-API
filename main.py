import uvicorn
from fastapi import FastAPI
from config.database import engine, Base
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from app.modules.knowledge_base import knowledge_base_route


import warnings

app = FastAPI(docs_url="/")

origins = ["*"]

warnings.filterwarnings("ignore")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


@app.get("/")
def welcome():
    return "Welcome to Agents AI Service"

app.include_router(knowledge_base_route.router)

add_pagination(app)

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000, log_level="info", reload=True)
    print("running")
