# Resume vs Job Description Matcher

## Overview
This project provides a Python script that uses AI to match resumes against multiple job descriptions (JDs). The script extracts key skills, evaluates the match, and provides actionable suggestions for improving resumes based on job requirements.

## Features
- Supports TXT, PDF, and DOCX formats for both resumes and job descriptions.
- Uses OpenAI's API for skill extraction, scoring, and reasoning.
- Outputs a CSV file with detailed analysis for each job description.

## Prerequisites
- Python 3.8 or higher
- Install required Python packages using `pip install -r requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deepseek_resume-job-matcher-multiJD-main
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your OpenAI API key:
     ```
     DEEPSEEK_API_KEY=<your-api-key>
     OPENAI_BASE_URL=https://api.deepseek.com
     OPENAI_MODEL=deepseek-chat
     ```

## Usage
Run the script with the following command:
```bash
python multi_jd_mode.py --resume <path-to-resume> --jds <path-to-jds-folder> --out <output-csv>
```

### Example
```bash
python multi_jd_mode.py --resume ./resumes/Jojo_resume.pdf --jds ./jds --out result.csv
```

### Arguments
- `--resume`: Path to the resume file (PDF, DOCX, or TXT).
- `--jds`: Path to the folder containing job descriptions.
- `--out`: Path to the output CSV file.

## Output
The script generates a CSV file with the following columns:
- `jd_file`: Name of the job description file.
- `overall_score`: Match score (0-100).
- `reasoning`: Explanation for the score.
- `resume_skills`: Extracted skills from the resume.
- `jd_skills`: Extracted skills from the job description.
- `missing_skills`: Skills missing in the resume.
- `suggestions`: Suggestions for improving the resume.

## Debugging
If you encounter issues:
1. Ensure the resume and job description files exist and are accessible.
2. Verify the `.env` file contains valid API keys.
3. Check the console output for detailed error messages.

## License
This project is licensed under the MIT License.
