"""
app.py — Streamlit web UI for the Resume Screening Agent

Run with:
    streamlit run app.py

The UI lets you:
  - Upload a job description (TXT/PDF/DOCX) or paste it directly
  - Upload multiple resume files
  - Run the full screening pipeline
  - View the ranked table with expandable per-candidate cards
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.embedder import batch_similarity
from agent.parser import extract_text
from agent.scorer import batch_score_resumes
from agent.ranker import rank_candidates, SEMANTIC_WEIGHT, LLM_WEIGHT

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main { background-color: #0f1117; }
        .stApp { background-color: #0f1117; }
        .score-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .strong-hire { background: #16a34a; color: white; }
        .hire        { background: #22c55e; color: white; }
        .maybe       { background: #ca8a04; color: white; }
        .no-hire     { background: #dc2626; color: white; }
        .metric-card {
            background: #1e2130;
            border: 1px solid #2d3148;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def recommendation_badge(rec: str) -> str:
    css_class = {
        "Strong Hire": "strong-hire",
        "Hire": "hire",
        "Maybe": "maybe",
        "No Hire": "no-hire",
    }.get(rec, "maybe")
    return f'<span class="score-badge {css_class}">{rec}</span>'


def score_bar(value: float, colour: str = "#6366f1") -> str:
    pct = int(value * 100)
    return (
        f'<div style="background:#2d3148;border-radius:6px;height:8px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{colour};border-radius:6px;"></div>'
        f'</div><small style="color:#94a3b8;">{value:.3f}</small>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a free key at console.groq.com",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    st.subheader("Score Weights")
    st.markdown(
        f"**Semantic similarity:** {int(SEMANTIC_WEIGHT*100)}%  \n"
        f"**LLM reasoning:** {int(LLM_WEIGHT*100)}%"
    )
    st.caption(
        "These weights are set in `agent/ranker.py` and can be tuned. "
        "LLM score is weighted higher because it captures depth of experience."
    )

    st.divider()
    st.caption("Resume Screening Agent · Rooman AI Challenge")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎯 Resume Screening Agent")
st.markdown(
    "Upload a **job description** and a set of **resumes** — the agent will rank "
    "candidates using semantic similarity + Groq LLM reasoning."
)

tab_upload, tab_results, tab_raw = st.tabs(["📂 Upload & Run", "📊 Results", "🗂️ Raw JSON"])

with tab_upload:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Description")
        jd_source = st.radio("JD source", ["Paste text", "Upload file"], horizontal=True)

        jd_text = ""
        if jd_source == "Paste text":
            jd_text = st.text_area(
                "Paste the job description here",
                height=300,
                placeholder="We are looking for a Data Scientist with ...",
            )
        else:
            jd_file = st.file_uploader("Upload JD", type=["txt", "pdf", "docx"])
            if jd_file:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(jd_file.name).suffix
                ) as tmp:
                    tmp.write(jd_file.read())
                    tmp_path = tmp.name
                jd_text = extract_text(tmp_path)
                os.unlink(tmp_path)
                if jd_text:
                    st.success(f"✓ Loaded JD ({len(jd_text)} chars)")
                else:
                    st.error("Could not extract text from JD file.")

    with col2:
        st.subheader("Resumes")
        resume_files = st.file_uploader(
            "Upload resume files",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            help="Upload 10+ resumes for best results.",
        )
        if resume_files:
            st.info(f"{len(resume_files)} file(s) uploaded.")

    st.divider()
    run_btn = st.button(
        "🚀 Run Screening Pipeline",
        type="primary",
        disabled=not (jd_text and resume_files and api_key),
    )

    if not api_key:
        st.warning("⚠️ Enter your Groq API key in the sidebar to enable screening.")

    if run_btn:
        # Save uploaded resumes to a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            resumes: dict[str, str] = {}
            for uf in resume_files:
                dest = Path(tmpdir) / uf.name
                dest.write_bytes(uf.read())
                text = extract_text(str(dest))
                resumes[uf.name] = text

            with st.spinner("Stage 1 — Computing semantic similarity ..."):
                sem_scores = batch_similarity(jd_text, resumes)

            progress_bar = st.progress(0)
            status_text = st.empty()

            llm_scores: dict[str, dict] = {}
            total = len(resumes)
            for idx, (fname, text) in enumerate(resumes.items(), start=1):
                status_text.text(f"Stage 2 — LLM scoring {idx}/{total}: {fname}")
                from agent.scorer import score_resume, _get_client
                client = _get_client()
                llm_scores[fname] = score_resume(jd_text, text, fname, client=client)
                progress_bar.progress(idx / total)
                time.sleep(0.5)

            status_text.empty()
            progress_bar.empty()

            # Combine and rank
            ranked: list[dict] = []
            for fname in resumes:
                sem = sem_scores.get(fname, 0.0)
                llm_d = llm_scores.get(fname, {})
                llm = llm_d.get("llm_score", 0.0)
                final = round(SEMANTIC_WEIGHT * sem + LLM_WEIGHT * llm, 4)
                ranked.append({
                    "candidate": fname,
                    "final_score": final,
                    "semantic_score": round(sem, 4),
                    "llm_score": round(llm, 4),
                    **llm_d,
                })

            ranked.sort(key=lambda c: c["final_score"], reverse=True)
            for i, c in enumerate(ranked, 1):
                c["rank"] = i

            st.session_state["ranked"] = ranked
            st.success("✅ Screening complete! Switch to the **Results** tab.")


with tab_results:
    ranked = st.session_state.get("ranked", [])

    if not ranked:
        st.info("Run the pipeline in the **Upload & Run** tab to see results.")
    else:
        # Summary metrics
        top = ranked[0]
        strong_hires = sum(1 for c in ranked if c.get("recommendation") == "Strong Hire")
        hires = sum(1 for c in ranked if c.get("recommendation") == "Hire")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates Screened", len(ranked))
        m2.metric("Strong Hires", strong_hires)
        m3.metric("Recommended Hires", strong_hires + hires)
        m4.metric("Top Score", f"{top['final_score']:.3f}")

        st.divider()
        st.subheader("Ranked Shortlist")

        # Ranked cards
        for c in ranked:
            rec = c.get("recommendation", "Maybe")
            with st.expander(
                f"#{c['rank']}  {c['candidate']}  —  {c['final_score']:.3f}",
                expanded=(c["rank"] == 1),
            ):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Final Score**\n\n{score_bar(c['final_score'], '#6366f1')}", unsafe_allow_html=True)
                c2.markdown(f"**Semantic**\n\n{score_bar(c['semantic_score'], '#0ea5e9')}", unsafe_allow_html=True)
                c3.markdown(f"**LLM Score**\n\n{score_bar(c['llm_score'], '#a855f7')}", unsafe_allow_html=True)

                st.markdown(
                    f"**Recommendation:** {recommendation_badge(rec)}&nbsp;&nbsp;"
                    f"**Education Fit:** {c.get('education_fit', '—')}&nbsp;&nbsp;"
                    f"**Experience:** {c.get('experience_years') or '—'} years",
                    unsafe_allow_html=True,
                )

                ca, cb = st.columns(2)
                ca.markdown(f"**✅ Matched Skills**\n\n{', '.join(c.get('matched_skills') or []) or 'None identified'}")
                cb.markdown(f"**❌ Missing Skills**\n\n{', '.join(c.get('missing_skills') or []) or 'None'}")

                st.markdown(f"**Strengths:** {c.get('strengths', '')}")
                st.markdown(f"**Gaps:** {c.get('gaps', '')}")


with tab_raw:
    ranked = st.session_state.get("ranked", [])
    if not ranked:
        st.info("Run the pipeline first.")
    else:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(ranked, indent=2),
            file_name="ranked_results.json",
            mime="application/json",
        )

        df = pd.DataFrame([
            {
                "rank": c["rank"],
                "candidate": c["candidate"],
                "final_score": c["final_score"],
                "semantic_score": c["semantic_score"],
                "llm_score": c["llm_score"],
                "recommendation": c.get("recommendation"),
                "experience_years": c.get("experience_years"),
                "matched_skills": "; ".join(c.get("matched_skills") or []),
                "missing_skills": "; ".join(c.get("missing_skills") or []),
            }
            for c in ranked
        ])

        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name="ranked_results.csv",
            mime="text/csv",
        )

        st.dataframe(df, use_container_width=True)
        st.json(ranked)
