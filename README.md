# MedInsight — Medical Report Explainer

An AI-powered system that reads medical PDF reports, explains findings 
in plain language, flags abnormal values, and answers follow-up questions.

## Demo
> Upload a blood report PDF → get instant plain-language explanations → ask follow-up questions

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Parser | pdfplumber + regex | Extracts structured test data |
| Neural Classifier | PyTorch MLP (3-layer) | Flags Normal/Low/High/Critical |
| RAG Pipeline | FAISS + sentence-transformers | Semantic search over report |
| Fine-tuned LLM | TinyLlama-1.1B + QLoRA | Medical Q&A answering |
| Agentic Loop | Custom ReAct-style agent | Routes queries intelligently |
| Backend | FastAPI | REST API with Swagger UI |
| Frontend | HTML/CSS/JS | Professional medical UI |

## Fine-tuning Details
- Base model: TinyLlama-1.1B-Chat-v1.0
- Dataset: PubMedQA (1000 medical Q&A pairs)
- Method: QLoRA (4-bit quantization, r=16, alpha=32)
- Platform: Google Colab T4 GPU (free tier)
- Trainable parameters: ~0.22% of total

## Neural Classifier
- Architecture: 3-layer MLP (5 → 32 → 16 → 4)
- Training accuracy: 90.8%
- Classes: Normal, Low, High, Critical
- Tests covered: 14 blood parameters

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/upload | Upload PDF, returns summary |
| GET | /api/summary/{report_id} | Check report status |
| POST | /api/ask/{report_id} | Ask a question |

## Setup & Run

```bash
git clone https://github.com/YOUR_USERNAME/medical-report-explainer
cd medical-report-explainer
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## Tech Stack
Python 3.11 · FastAPI · PyTorch · HuggingFace Transformers · 
PEFT · TRL · FAISS · sentence-transformers · pdfplumber
