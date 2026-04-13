import pdfplumber
import re

KNOWN_TESTS = [
    "Hemoglobin (Hb)", "Total RBC count", "Packed Cell Volume (PCV)",
    "Mean Corpuscular Volume (MCV)", "MCH", "MCHC", "RDW",
    "Total WBC count", "Neutrophils", "Lymphocytes", "Eosinophils",
    "Monocytes", "Basophils", "Platelet Count"
]

FLAGS = ["Low", "High", "Borderline", "Critical"]

def read_pdf(file_path: str) -> str:
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def parse_report(file_path: str) -> dict:
    text = read_pdf(file_path)
    lines = text.split("\n")
    results = []

    for line in lines:
        line = line.strip()
        line = line.replace("Calculated", "").strip()
        matched_test = None

        for test in KNOWN_TESTS:
            if line.startswith(test):
                matched_test = test
                break

        if not matched_test:
            continue

        remainder = line[len(matched_test):].strip()
        tokens = remainder.split()

        if not tokens:
            continue

        try:
            value = float(tokens[0])
        except ValueError:
            continue

        flag = None
        ref_start_idx = 1

        if len(tokens) > 1 and tokens[1] in FLAGS:
            flag = tokens[1]
            ref_start_idx = 2

        ref_range = None
        unit = None

        ref_pattern = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", remainder)
        if ref_pattern:
            ref_range = {
                "low": float(ref_pattern.group(1)),
                "high": float(ref_pattern.group(2))
            }

        unit_pattern = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s+(\S+)$", remainder)
        if unit_pattern:
            unit = unit_pattern.group(3)

        results.append({
            "test": matched_test,
            "value": value,
            "flag": flag,
            "reference_range": ref_range,
            "unit": unit
        })

    patient_info = extract_patient_info(text)

    return {
        "patient": patient_info,
        "results": results
    }


def extract_patient_info(text: str) -> dict:
    info = {}

    name_match = re.search(r"^([A-Z][a-z]+(?:\s[A-Z]\.?\s?[A-Z][a-z]+)+)", text, re.MULTILINE)
    if name_match:
        info["name"] = name_match.group(1)

    age_match = re.search(r"Age\s*:\s*(\d+)\s*Years", text)
    if age_match:
        info["age"] = int(age_match.group(1))

    sex_match = re.search(r"Sex\s*:\s*(\w+)", text)
    if sex_match:
        info["sex"] = sex_match.group(1)

    return info


if __name__ == "__main__":
    import json
    report = parse_report("/Users/adismacbook/Desktop/medical-report-explainer/data/uploads/CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf")
    print(json.dumps(report, indent=2))