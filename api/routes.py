from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pipeline.agent import process_report, answer_question
from pydantic import BaseModel
import shutil
import uuid
import os

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str


@router.post("/upload")
async def upload_report(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    report_id = str(uuid.uuid4())[:8]
    save_path = f"data/uploads/{report_id}.pdf"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        report = process_report(save_path, report_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "report_id": report_id,
        "patient": report["patient"],
        "summary": report["summary"],
        "message": "Report processed successfully"
    }


@router.get("/summary/{report_id}")
async def get_summary(report_id: str):
    index_path = f"data/indexes/{report_id}"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return JSONResponse(content={
        "report_id": report_id,
        "message": "Use /upload to get full summary or /ask to ask questions"
    })


@router.post("/ask/{report_id}")
async def ask_question(report_id: str, body: QuestionRequest):
    index_path = f"data/indexes/{report_id}"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Report not found. Upload it first.")

    try:
        result = answer_question(report_id, body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result