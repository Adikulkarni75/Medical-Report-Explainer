from pipeline.pdf_parser import parse_report, read_pdf
from pipeline.classifier import classify_value, REFERENCE_RANGES
from pipeline.rag_pipeline import build_index, retrieve
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

MODEL_PATH = "models/tinyllama-medical-adapter"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_llm = None
_tokenizer = None


def load_llm():
    global _llm, _tokenizer
    if _llm is not None:
        return _llm, _tokenizer

    print("Loading LLM...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        torch_dtype=torch.float32
    )

    _llm = PeftModel.from_pretrained(base, MODEL_PATH)
    _llm.eval()
    _llm.config.use_cache = True
    print("LLM loaded!")
    return _llm, _tokenizer


def ask_llm(question: str, context: str) -> str:
    model, tokenizer = load_llm()

    prompt = f"""### Instruction:
You are a medical assistant. Answer the following medical question clearly and in simple language a patient can understand.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split("### Answer:")[-1].strip()
    return answer


def generate_summary(parsed_report: dict) -> list:
    summary = []
    for item in parsed_report["results"]:
        test = item["test"]
        value = item["value"]
        unit = item.get("unit", "")
        ref = item.get("reference_range")
        pdf_flag = item.get("flag")

        classification = classify_value(test, value)
        status = classification["status"]
        confidence = classification["confidence"]

        if pdf_flag and status == "Normal":
            status = pdf_flag

        ref_str = ""
        if ref:
            ref_str = f"{ref['low']} - {ref['high']} {unit}"

        summary.append({
            "test": test,
            "value": value,
            "unit": unit,
            "status": status,
            "confidence": confidence,
            "reference_range": ref_str,
        })
    return summary


def answer_question(report_id: str, question: str) -> dict:
    test_keywords = list(REFERENCE_RANGES.keys())
    question_lower = question.lower()

    use_rag = any(k.lower() in question_lower for k in test_keywords) or \
              any(w in question_lower for w in ["value", "result", "level", "count", "report"])

    if use_rag:
        chunks = retrieve(report_id, question, top_k=3)
        context = "\n".join([c["chunk"] for c in chunks])
        source = "report"
    else:
        context = "General medical knowledge about blood tests and lab results."
        source = "llm"

    answer = ask_llm(question, context)

    return {
        "question": question,
        "answer": answer,
        "source": source
    }


def process_report(pdf_path: str, report_id: str) -> dict:
    print(f"Parsing PDF: {pdf_path}")
    parsed = parse_report(pdf_path)

    print("Building RAG index...")
    text = read_pdf(pdf_path)
    build_index(report_id, text)

    print("Generating summary with classifier...")
    summary = generate_summary(parsed)

    return {
        "report_id": report_id,
        "patient": parsed["patient"],
        "summary": summary
    }


if __name__ == "__main__":
    import json

    report = process_report("/Users/adismacbook/Desktop/medical-report-explainer/data/uploads/CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf", "test_001")

    print("\nPatient:", json.dumps(report["patient"], indent=2))
    print("\nTest Summary:")
    for item in report["summary"]:
        flag = f"[{item['status']}]" if item['status'] != "Normal" else ""
        print(f"  {item['test']}: {item['value']} {item['unit']} {flag}")

    print("\nTesting Q&A (RAG only, no LLM load):")
    from pipeline.rag_pipeline import retrieve
    chunks = retrieve("test_001", "what is the hemoglobin level?", top_k=2)
    print("Retrieved context:")
    for c in chunks:
        print(f"  → {c['chunk'][:100]}...")