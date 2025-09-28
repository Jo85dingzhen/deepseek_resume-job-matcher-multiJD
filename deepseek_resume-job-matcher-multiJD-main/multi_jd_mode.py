"""
multi_jd_mode_llm_only.py — One Resume vs Many JDs (DeepSeek-enabled, LLM-centric)

Usage:
    pip install -r requirements.txt langchain openai tiktoken
    # prepare .env with DEEPSEEK_API_KEY=...
    python multi_jd_mode_llm_only.py --resume ./resumes/Jojo_resume.pdf --jds ./jds --out result_llm_centric.csv

This version relies entirely on LLMs for skill extraction, scoring, and providing reasoning/suggestions.
It aims to be robust to diverse CV formats by letting the LLM handle the parsing and interpretation.
"""
from __future__ import annotations
import os, re, json, argparse
from typing import Dict, Any, List
import pandas as pd

# --- dotenv (optional) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- readers ---
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(path) or ""

def read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_any(path: str) -> str:
    # Ensure the path is absolute
    path = os.path.abspath(path)
    print(f"Resolved absolute path: {path}")  # Log the resolved path for debugging

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".docx", ".doc"):
        return read_docx(path)
    return read_txt(path)

# --- Environment Validation ---
def validate_env_variables():
    """Ensure required environment variables are set."""
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY or OPENAI_API_KEY not found in environment variables.")
    return api_key

# --- Enhanced LLM Scoring Core ---
def match_resume_to_jd_llm_only(resume_text: str, jd_text: str):
    api_key = validate_env_variables()

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")

    prompt = (
        "You are an expert career coach and resume evaluator. Your task is to analyze a candidate's resume "
        "against a specific job description (JD). You need to perform the following steps:\n\n"
        "1.  **Extract Key Skills:** Identify and list the most prominent technical and soft skills from both the "
        "    RESUME and the JOB DESCRIPTION separately.\n"
        "2.  **Evaluate Match (Subscores):** Assign subscores (0-100) for the following categories:\n"
        "    - Skills\n"
        "    - Experience\n"
        "    - Education\n"
        "    - Keywords\n"
        "    Provide a short explanation for each subscore.\n"
        "3.  **Compute Overall Score:** The overall score will be calculated by the caller using fixed weights:\n"
        "    - Skills: 40%\n"
        "    - Experience: 30%\n"
        "    - Education: 15%\n"
        "    - Keywords: 15%\n"
        "4.  **Provide Detailed Reasoning:** Explain *why* you assigned the subscores. Highlight the strengths of the "
        "    resume relative to the JD and point out key areas of weakness or missing requirements. Be specific.\n"
        "5.  **Identify Top Missing Skills/Experiences:** List 3-5 crucial skills or experiences explicitly mentioned "
        "    in the JD that are clearly missing or underdeveloped in the resume.\n"
        "6.  **Propose Resume Edits/Suggestions:** Based on the identified weaknesses, provide 3-5 concrete, "
        "    resume-ready bullet point suggestions (action verbs, quantifiable results, max 28 words per bullet) "
        "    for the candidate to improve their resume for *this specific JD*. Focus on bridging the gaps.\n\n"
        "**Your output MUST be a JSON object with the following keys:**\n"
        "   - `resume_skills`: [List of extracted skills from resume]\n"
        "   - `jd_skills`: [List of extracted skills from JD]\n"
        "   - `subscores`: {\"skills\": INT, \"experience\": INT, \"education\": INT, \"keywords\": INT}\n"
        "   - `subscores_explanations`: {\"skills\": \"short string\", \"experience\": \"short string\", \"education\": \"short string\", \"keywords\": \"short string\"}\n"
        "   - `top_missing_skills`: [List of 3-5 missing critical skills/experiences]\n"
        "   - `suggestions`: [List of 3-5 resume bullet point suggestions]\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text}\n"
        "\nProvide only the JSON output."
    )

    for attempt in range(3):  # Retry logic
        try:
            print(f"Attempt {attempt + 1}: Sending request to LLM...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temperature for stability
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            raw_content = resp.choices[0].message.content.strip()
            print(f"Raw response: {raw_content}")
            data = json.loads(raw_content)

            # Calculate overall score locally
            weights = {"skills": 0.4, "experience": 0.3, "education": 0.15, "keywords": 0.15}
            subscores = data["subscores"]
            overall_score = sum(subscores[k] * weights[k] for k in weights)
            data["overall_score"] = int(round(overall_score))

            return data
        except Exception as e:
            print(f"Error during LLM request: {e}")
            if attempt < 2:  # Retry up to 3 times
                print("Retrying...")
                continue
            else:
                print("Max retries reached. Returning error.")
                return {
                    "error": str(e),
                    "resume_skills": [],
                    "jd_skills": [],
                    "subscores": {"skills": 0, "experience": 0, "education": 0, "keywords": 0},
                    "subscores_explanations": {},
                    "overall_score": 0,
                    "reasoning": f"LLM processing failed: {e}",
                    "top_missing_skills": [],
                    "suggestions": []
                }

def main():
    ap = argparse.ArgumentParser(description="One Resume vs Many JDs (LLM-Centric) → CSV")
    ap.add_argument("--resume", required=True, help="Path to resume (pdf/docx/txt)")
    ap.add_argument("--jds", required=True, help="Folder of JDs (pdf/docx/txt)")
    ap.add_argument("--out", default="resume_results.csv", help="Output CSV")
    args = ap.parse_args()

    resume_text = read_any(args.resume)
    rows = []
    
    print("Starting LLM-centric evaluation...")
    for name in sorted(os.listdir(args.jds)):
        path = os.path.join(args.jds, name)
        if os.path.isdir(path):
            continue
        
        print(f"Processing JD: {name}")
        try:
            jd_text = read_any(path)
            print(f"Processing JD: {name}")

            # Call the LLM-centric matching function
            try:
                print(f"Sending request to LLM for JD: {name}")
                prompt = (
                    "You are an expert career coach and resume evaluator. Your task is to analyze a candidate's resume "
                    "against a specific job description (JD). You need to perform the following steps:\n\n"
                    "1.  **Extract Key Skills:** Identify and list the most prominent technical and soft skills from both the "
                    "    RESUME and the JOB DESCRIPTION separately.\n"
                    "2.  **Evaluate Match (Subscores):** Assign subscores (0-100) for the following categories:\n"
                    "    - Skills\n"
                    "    - Experience\n"
                    "    - Education\n"
                    "    - Keywords\n"
                    "    Provide a short explanation for each subscore.\n"
                    "3.  **Compute Overall Score:** The overall score will be calculated by the caller using fixed weights:\n"
                    "    - Skills: 40%\n"
                    "    - Experience: 30%\n"
                    "    - Education: 15%\n"
                    "    - Keywords: 15%\n"
                    "4.  **Provide Detailed Reasoning:** Explain *why* you assigned the subscores. Highlight the strengths of the "
                    "    resume relative to the JD and point out key areas of weakness or missing requirements. Be specific.\n"
                    "5.  **Identify Top Missing Skills/Experiences:** List 3-5 crucial skills or experiences explicitly mentioned "
                    "    in the JD that are clearly missing or underdeveloped in the resume.\n"
                    "6.  **Propose Resume Edits/Suggestions:** Based on the identified weaknesses, provide 3-5 concrete, "
                    "    resume-ready bullet point suggestions (action verbs, quantifiable results, max 28 words per bullet) "
                    "    for the candidate to improve their resume for *this specific JD*. Focus on bridging the gaps.\n\n"
                    "**Your output MUST be a JSON object with the following keys:**\n"
                    "   - `resume_skills`: [List of extracted skills from resume]\n"
                    "   - `jd_skills`: [List of extracted skills from JD]\n"
                    "   - `subscores`: {\"skills\": INT, \"experience\": INT, \"education\": INT, \"keywords\": INT}\n"
                    "   - `subscores_explanations`: {\"skills\": \"short string\", \"experience\": \"short string\", \"education\": \"short string\", \"keywords\": \"short string\"}\n"
                    "   - `top_missing_skills`: [List of 3-5 missing critical skills/experiences]\n"
                    "   - `suggestions`: [List of 3-5 resume bullet point suggestions]\n\n"
                    f"--- RESUME ---\n{resume_text}\n\n"
                    f"--- JOB DESCRIPTION ---\n{jd_text}\n"
                    "\nProvide only the JSON output."
                )
                print(f"Prompt: {prompt}")  # Log the prompt for debugging

                resp = match_resume_to_jd_llm_only(resume_text, jd_text)
                print(f"LLM Response: {resp}")  # Log the full response

                if "error" in resp:
                    rows.append({
                        "jd_file": name,
                        "overall_score": -1,
                        "reasoning": resp.get("reasoning", "Error during processing"),
                        "resume_skills": "",
                        "jd_skills": "",
                        "missing_skills": resp.get("error", "Unknown error"),
                        "suggestions": ""
                    })
                    continue

                # Validate response structure
                missing_keys = [key for key in ["resume_skills", "jd_skills", "subscores", "reasoning", "top_missing_skills", "suggestions"] if key not in resp]
                if missing_keys:
                    print(f"Invalid response structure for JD: {name}. Missing keys: {missing_keys}. Response: {resp}")
                    rows.append({
                        "jd_file": name,
                        "overall_score": resp.get("overall_score", -1),
                        "reasoning": resp.get("reasoning", "Invalid response structure"),
                        "resume_skills": ";".join(resp.get("resume_skills", [])),
                        "jd_skills": ";".join(resp.get("jd_skills", [])),
                        "missing_skills": ";".join(resp.get("top_missing_skills", [])),
                        "suggestions": "\n".join(resp.get("suggestions", []))
                    })
                    continue

                rows.append({
                    "jd_file": name,
                    "overall_score": resp["overall_score"],
                    "reasoning": resp["reasoning"],
                    "resume_skills": ";".join(resp["resume_skills"]),
                    "jd_skills": ";".join(resp["jd_skills"]),
                    "missing_skills": ";".join(resp["top_missing_skills"]),
                    "suggestions": "\n".join(resp["suggestions"])
                })
            except Exception as e:
                print(f"Error processing JD: {name}. Exception: {e}")
                rows.append({
                    "jd_file": name,
                    "overall_score": -1,
                    "reasoning": f"Local processing error: {e}",
                    "resume_skills": "",
                    "jd_skills": "",
                    "missing_skills": f"Local error: {e}",
                    "suggestions": ""
                })
        except Exception as e:
            print(f"Error reading JD file {name}: {e}")
            rows.append({
                "jd_file": name,
                "overall_score": -1,
                "reasoning": f"File reading error: {e}",
                "resume_skills": "",
                "jd_skills": "",
                "missing_skills": f"File error: {e}",
                "suggestions": ""
            })

    try:
        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False)
        print(f"\nEvaluation complete. Results saved to: {os.path.abspath(args.out)}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        raise e


if __name__ == "__main__":
    main()