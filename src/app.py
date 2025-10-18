from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Alzheimer Predictor API",
    description="Lightweight API for health checks and prediction endpoints for the Alzheimer predictor project.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NumReq(BaseModel):
    request_info: str | None = None

@app.get("/number")
def get_number():
    return {"number": 424234}

@app.post("/number")
def post_number(req: NumReq):
    return {"number": 424234234234}
