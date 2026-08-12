"""
ranker.py — Combine scores and produce the final ranked shortlist

This module ties together the two scoring stages:
  Stage 1: Semantic similarity from embedder.py  (weight: 40%)
  Stage 2: LLM reasoning score from scorer.py    (weight: 60%)

Final Score = 0.4 × semantic_score + 0.6 × llm_score

Weighting rationale:
  - LLM score is weighted higher because it captures depth of experience,
    education relevance, and contextual fit that cosine similarity misses.
  - Semantic score still matters as a fast, unbiased first-pass filter.

Output: a list of dicts sorted descending by final_score, exported to
        both JSON (for programmatic use) and CSV (for HR teams).
"""

import csv
import json
import logging
import os
from pathlib import Path

from agent.embedder import batch_similarity
from agent.scorer import batch_score_resumes

logger = logging.getLogger(__name__)

# Score weighting constants — easy to tune
SEMANTIC_WEIGHT = 0.4
LLM_WEIGHT = 0.6

# Recommendation label ordering (for tiebreaking, highest first)
RECOMMENDATION_ORDER = {
    "Strong Hire": 4,
    "Hire": 3,
    "Maybe": 2,
    "No Hire": 1,
}


def rank_candidates(
    jd_text: str,
    resumes: dict[str, str],
    output_dir: str | None = None,
) -> list[dict]:
    """
    Full pipeline: embed → LLM score → combine → sort → optionally export.

    Args:
        jd_text:     Full text of the job description.
        resumes:     Dict mapping filename → resume text.
        output_dir:  If provided, exports ranked_results.json and .csv here.

    Returns:
        List of candidate dicts, sorted descending by final_score.
    """
    # ── Stage 1: Semantic similarity ──────────────────────────────────────────
    logger.info("Stage 1 — Computing semantic similarity for %d resumes ...", len(resumes))
    semantic_scores = batch_similarity(jd_text, resumes)

    # ── Stage 2: LLM scoring ──────────────────────────────────────────────────
    logger.info("Stage 2 — LLM scoring via Groq ...")
    llm_scores = batch_score_resumes(jd_text, resumes)

    # ── Combine scores ────────────────────────────────────────────────────────
    ranked: list[dict] = []

    for filename in resumes:
        sem_score = semantic_scores.get(filename, 0.0)
        llm_data = llm_scores.get(filename, {})
        llm_score = llm_data.get("llm_score", 0.0)

        final_score = round(
            SEMANTIC_WEIGHT * sem_score + LLM_WEIGHT * llm_score, 4
        )

        candidate = {
            "candidate": filename,
            "final_score": final_score,
            "semantic_score": round(sem_score, 4),
            "llm_score": round(llm_score, 4),
            "llm_raw_score": llm_data.get("llm_raw_score", 0.0),
            "matched_skills": llm_data.get("matched_skills", []),
            "missing_skills": llm_data.get("missing_skills", []),
            "experience_years": llm_data.get("experience_years"),
            "education_fit": llm_data.get("education_fit", "Not Stated"),
            "strengths": llm_data.get("strengths", ""),
            "gaps": llm_data.get("gaps", ""),
            "recommendation": llm_data.get("recommendation", "Maybe"),
        }
        ranked.append(candidate)

    # ── Sort: primary = final_score desc, secondary = recommendation label desc ──
    ranked.sort(
        key=lambda c: (
            c["final_score"],
            RECOMMENDATION_ORDER.get(c["recommendation"], 0),
        ),
        reverse=True,
    )

    # ── Assign rank positions ─────────────────────────────────────────────────
    for i, candidate in enumerate(ranked, start=1):
        candidate["rank"] = i

    # ── Export ────────────────────────────────────────────────────────────────
    if output_dir:
        _export_results(ranked, output_dir)

    logger.info("Ranking complete. Top candidate: %s (%.4f)", ranked[0]["candidate"], ranked[0]["final_score"])
    return ranked


def _export_results(ranked: list[dict], output_dir: str) -> None:
    """
    Write ranked results to JSON and CSV files.

    Args:
        ranked:     Sorted list of candidate dicts.
        output_dir: Directory to write output files into.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "ranked_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)
    logger.info("JSON results written to '%s'.", json_path)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "ranked_results.csv")
    csv_fields = [
        "rank",
        "candidate",
        "final_score",
        "semantic_score",
        "llm_score",
        "llm_raw_score",
        "experience_years",
        "education_fit",
        "recommendation",
        "matched_skills",
        "missing_skills",
        "strengths",
        "gaps",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            # Flatten list fields to semicolon-separated strings for CSV
            row_copy = dict(row)
            row_copy["matched_skills"] = "; ".join(row_copy.get("matched_skills") or [])
            row_copy["missing_skills"] = "; ".join(row_copy.get("missing_skills") or [])
            writer.writerow(row_copy)

    logger.info("CSV results written to '%s'.", csv_path)
