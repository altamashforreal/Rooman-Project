# Resume Screening Agent

An AI-powered resume screening agent built for the Rooman AI Challenge. It parses multiple resumes (PDF/DOCX/TXT), scores each one against a Job Description using a **two-stage NLP pipeline** (semantic embeddings + LLM reasoning), and outputs a ranked shortlist with per-candidate explanations.

---

## How It Works

**Stage 1 — Semantic Similarity**
Each resume and the JD are embedded using `sentence-transformers` (all-MiniLM-L6-v2). Cosine similarity gives an objective overlap score between 0 and 1.

**Stage 2 — LLM Re-ranking**
The Groq API (llama-3.3-70b-versatile) reads each resume alongside the JD and returns a structured JSON score with strengths, skill gaps, and a hiring recommendation.

**Final Score = 0.4 × semantic_score + 0.6 × llm_score**
The LLM score is weighted higher because it captures reasoning about experience depth and context — things pure similarity misses.

---

## Project Structure

```
resume-screening-agent/
├── agent/
│   ├── __init__.py
│   ├── parser.py       # PDF / DOCX / TXT text extraction
│   ├── embedder.py     # sentence-transformers cosine similarity
│   ├── scorer.py       # Groq LLM structured scoring
│   └── ranker.py       # combine scores → ranked output
├── data/
│   ├── job_description.txt
│   └── resumes/        # 10 sample resumes
├── outputs/            # generated after running
│   ├── ranked_results.json
│   └── ranked_results.csv
├── main.py             # CLI entrypoint
├── app.py              # Streamlit web UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd resume-screening-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run. This is automatic.

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and paste your Groq API key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free key at [console.groq.com](https://console.groq.com).

---

## Running the Agent

### CLI (recommended for reviewers)

```bash
python main.py --jd data/job_description.txt --resumes data/resumes/ --output outputs/
```

**Output:**
- Ranked table printed to terminal
- `outputs/ranked_results.json`
- `outputs/ranked_results.csv`

**Example terminal output:**

```
╔══════════════════════════════════════════════════════════════╗
║           RESUME SCREENING AGENT — RANKED RESULTS           ║
╚══════════════════════════════════════════════════════════════╝

 Rank  Candidate              Final Score  Semantic   LLM Score  Recommendation
 1     priya_sharma.txt       0.847        0.81       0.87       Strong Hire
 2     arjun_mehta.pdf        0.791        0.76       0.81       Hire
 3     sarah_johnson.docx     0.754        0.74       0.76       Hire
 ...
```

### Streamlit Web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — upload a JD and resume folder, see ranked results in a visual table with expandable reasoning cards per candidate.

---

## Sample Inputs / Outputs

**Job Description:** Data Scientist role (see `data/job_description.txt`)

**10 sample resumes** in `data/resumes/` covering a range of experience levels and skill overlaps.

**Sample output** (from `outputs/ranked_results.json`):

```json
[
  {
    "rank": 1,
    "candidate": "priya_sharma.txt",
    "final_score": 0.847,
    "semantic_score": 0.81,
    "llm_score": 0.87,
    "matched_skills": ["Python", "Machine Learning", "SQL", "TensorFlow", "pandas"],
    "missing_skills": ["Spark", "Kubernetes"],
    "strengths": "5 years of hands-on ML experience with strong publication record. Excellent Python and deep learning skills.",
    "gaps": "No big data (Spark) experience. Limited cloud deployment exposure.",
    "recommendation": "Strong Hire"
  }
]
```

---

## Design Choices & Tradeoffs

| Decision | Choice | Reasoning |
|---|---|---|
| Embeddings model | `all-MiniLM-L6-v2` | Small, fast, free, runs locally — no API cost |
| LLM provider | Groq (llama-3.3-70b) | Free tier, very fast inference, strong reasoning |
| Score weighting | 40% semantic + 60% LLM | LLM captures context; embeddings capture coverage |
| PDF parsing | `pdfplumber` | Reliable for text-based PDFs; no OCR needed |
| Output format | JSON + CSV | JSON for programmatic use, CSV for HR teams |
| UI | Streamlit | Minimal setup, visual enough to demo immediately |

### Known Limitations
- **Scanned PDFs** (image-only) are not supported — only text-based PDFs
- **Score calibration** depends on JD specificity; very short JDs produce noisier scores
- **LLM rate limits** on Groq's free tier may slow processing for 20+ resumes
- **Bias awareness:** LLM scores should be treated as a decision-support tool, not a final hiring decision

### What I'd improve with more time
- Add OCR support for scanned documents using `pytesseract`
- Use a fine-tuned HR-domain embedding model for better domain-specific matching
- Add a feedback loop — let users mark scores as helpful/not helpful and retrain weights
- Store results in SQLite with historical run tracking

---

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | Groq LLM API client |
| `sentence-transformers` | Local semantic embeddings |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `streamlit` | Web UI |
| `pandas` | CSV export and tabular display |
| `scikit-learn` | Cosine similarity computation |
| `rich` | Pretty terminal output |
| `python-dotenv` | API key management |

---

## Author

Built for the Rooman AI Challenge — Junior AI Research Associate Selection Round.
