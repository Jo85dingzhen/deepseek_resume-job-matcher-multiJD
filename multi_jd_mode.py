"""
multi_jd_mode.py — One Resume vs Many JDs (DeepSeek-enabled)

Usage (standalone):
    pip install -r requirements.txt
    # prepare .env with DEEPSEEK_API_KEY=...
    python multi_jd_mode.py --resume ./resumes/Jojo_resume.pdf --jds ./jds --out result_resume_vs_jds.csv --use-llm

Usage (as a drop-in to your existing repo):
    - Put this file at project root, ensure requirements installed (see requirements.txt in this pack).
    - It does NOT modify your original scripts; it's an extra mode.
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
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".docx", ".doc"):
        return read_docx(path)
    return read_txt(path)

# --- scoring core ---
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:
    ENGLISH_STOP_WORDS = set()

def normalize_text(t: str) -> str:
    import re
    t = t.lower()
    t = re.sub(r"[^a-z0-9\-\+\#\.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def token_set(text: str) -> set:
    toks = [w for w in re.split(r"\W+", text.lower()) if w and w not in ENGLISH_STOP_WORDS]
    return set(toks)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic_similarity": 0.40,
    "skill_overlap": 0.30,
    "experience_alignment": 0.15,
    "education_match": 0.10,
    "keyword_coverage": 0.05,
}

SKILL_SETS = {
    "core": [
        "python","java","c++","sql","mysql","postgres","r","git","rest","api",
        "cloud","aws","azure","gcp","docker","kubernetes","linux","pandas","numpy",
        "scikit-learn","tableau","powerbi","excel","vba"
    ],
    "fintech": [
        "payments","risk","aml","kyc","fraud detection","credit scoring","trading",
        "market data","bloomberg","fix protocol","derivatives","valuation","compliance"
    ],
    "swe": [
        "system design","microservices","distributed systems","data structures","algorithms",
        "message queue","kafka","redis","mongodb","grpc","graphql","cicd","testing","unit test"
    ]
}

_EMBEDDER = None
def load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER

def embed_sim(a: str, b: str) -> float:
    from sentence_transformers import util
    model = load_embedder()
    e1 = model.encode([a], convert_to_tensor=True, normalize_embeddings=True)
    e2 = model.encode([b], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(e1, e2).cpu().numpy()[0][0])

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def extract_skills(text: str):
    txt = normalize_text(text)
    found = set()
    for bucket in SKILL_SETS.values():
        for s in bucket:
            if s in txt:
                found.add(s)
    return sorted(found)

def keyword_coverage(resume: str, jd: str) -> float:
    rset = token_set(resume)
    jset = token_set(jd)
    if not jset:
        return 0.0
    return len(rset & jset) / len(jset)

def score_education(resume: str, jd: str) -> float:
    r = normalize_text(resume)
    j = normalize_text(jd)
    want_master = "master" in j
    want_bachelor = any(k in j for k in ["bachelor","bs","b.s."])
    has_master = any(k in r for k in ["master of","m.s.","ms "])
    has_bachelor = any(k in r for k in ["bachelor of","b.s.","bs "])
    if want_master and has_master:
        return 1.0
    if want_bachelor and (has_bachelor or has_master):
        return 1.0
    if not want_master and not want_bachelor:
        return 0.7
    return 0.0

def score_experience_alignment(resume: str, jd: str) -> float:
    verbs = ["built","developed","designed","implemented","deployed","optimized","led","managed","analyzed"]
    r = normalize_text(resume)
    j = normalize_text(jd)
    verb_score = sum(1 for v in verbs if v in r and v in j) / max(1, len(verbs))
    r_sk = set(extract_skills(resume))
    j_sk = set(extract_skills(jd))
    return 0.5*verb_score + 0.5*jaccard(r_sk, j_sk)

def top_missing_skills(resume: str, jd: str, k: int = 8):
    r_sk = set(extract_skills(resume))
    j_sk = set(extract_skills(jd))
    return sorted(list(j_sk - r_sk))[:k]

def match_resume_to_jd(resume: str, jd: str, weights: Dict[str,float]=None, use_llm: bool=False):
    weights = weights or DEFAULT_WEIGHTS
    resume = resume or ""
    jd = jd or ""

    sem = embed_sim(resume, jd)
    r_sk = set(extract_skills(resume))
    j_sk = set(extract_skills(jd))
    sk_ol = jaccard(r_sk, j_sk)
    exp = score_experience_alignment(resume, jd)
    edu = score_education(resume, jd)
    kw = keyword_coverage(resume, jd)

    parts = {
        "semantic_similarity": float(sem),
        "skill_overlap": float(sk_ol),
        "experience_alignment": float(exp),
        "education_match": float(edu),
        "keyword_coverage": float(kw)
    }
    score = sum(parts[k]*weights.get(k,0.0) for k in parts)
    missing = top_missing_skills(resume, jd)

    suggestions = []
    if use_llm:
        # DeepSeek via OpenAI SDK
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL","https://api.deepseek.com"))
                model = os.getenv("OPENAI_MODEL","deepseek-chat")
                prompt = (
                    "You are a helpful career coach. You will receive a candidate's resume and a job description (JD).\n"
                    "1) Identify the top missing or weak skills/experiences relative to the JD.\n"
                    "2) Propose 5-8 resume-ready bullet edits (action verbs, <28 words, use metrics).\n\n"
                    f"RESUME:\n{resume}\n\nJD:\n{jd}\n\n"
                    "Answer in JSON with keys: \"missing_skills\", \"suggestions\"."
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.7
                )
                import json as _json
                raw = resp.choices[0].message.content.strip()
                data = _json.loads(raw)
                suggestions = data.get("suggestions", [])
                ms = data.get("missing_skills")
                if ms: missing = ms
            except Exception:
                pass

    return {
        "overall_score": round(float(score), 4),
        "breakdown": {k: round(float(v),4) for k,v in parts.items()},
        "top_missing_skills": missing,
        "suggestions": suggestions,
        "debug": {
            "skills_found": sorted(list(r_sk)),
            "jd_skills": sorted(list(j_sk))
        }
    }

def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description="One Resume vs Many JDs → CSV")
    ap.add_argument("--resume", required=True, help="Path to resume (pdf/docx/txt)")
    ap.add_argument("--jds", required=True, help="Folder of JDs (pdf/docx/txt)")
    ap.add_argument("--out", default="resume_vs_jds.csv", help="Output CSV")
    ap.add_argument("--use-llm", action="store_true", help="Use DeepSeek suggestions (requires DEEPSEEK_API_KEY)")
    args = ap.parse_args()

    resume_text = read_any(args.resume)
    rows = []
    for name in sorted(os.listdir(args.jds)):
        path = os.path.join(args.jds, name)
        if os.path.isdir(path):
            continue
        try:
            jd_text = read_any(path)
            res = match_resume_to_jd(resume_text, jd_text, use_llm=args.use_llm)
            rows.append({
                "jd_file": name,
                "overall_score": res["overall_score"],
                **res["breakdown"],
                "missing_skills": ";".join(res["top_missing_skills"])
            })
        except Exception as e:
            rows.append({
                "jd_file": name,
                "overall_score": -1,
                "semantic_similarity": 0,
                "skill_overlap": 0,
                "experience_alignment": 0,
                "education_match": 0,
                "keyword_coverage": 0,
                "missing_skills": f"ERROR: {e}"
            })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
