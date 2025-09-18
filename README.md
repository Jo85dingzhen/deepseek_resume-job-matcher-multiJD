# One Resume vs Many JDs (DeepSeek-enabled)

## Quickstart
```bash
pip install -r requirements.txt
# Next Step
python multi_jd_mode.py --resume ./resumes/Jojo_resume.pdf --jds ./jds --out result.csv --use-llm
```

- Supports TXT / PDF / DOCX for both resume and JDs
- Outputs a CSV with per-JD scores + missing skills
- Uses DeepSeek via OpenAI SDK (base_url=https://api.deepseek.com)
