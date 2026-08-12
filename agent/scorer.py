"""
scorer.py — LLM-based resume scoring via Groq

This is Stage 2 of the pipeline. After computing semantic similarity in
embedder.py, we send each resume + JD to the Groq API (llama-3.3-70b)
and ask it to return a structured JSON score with:

  - A numeric score (0–10)
  - Matched skills
  - Missing/gap skills
  - Candidate strengths (narrative)
  - Candidate gaps (narrative)
  - A hiring recommendation label

Design decision — why LLM scoring matters:
  Cosine similarity catches keyword/concept overlap but can't distinguish
  "5 years of ML" from "one ML course". The LLM reads context and reasons
  about experience depth, role fit, and education relevance. That's why
  we weight LLM score at 60% in the final composite.

Rate limiting:
  Groq's free tier allows ~30 RPM. For 10 resumes this is fine.
  For 20+ resumes, add a short sleep between calls.
"""

import json
import logging
import os
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Groq model to use — llama-3.3-70b is free and handles structured output well
GROQ_MODEL = "llama-3.3-70b-versatile"

# Retry config for transient API errors
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def _get_client() -> Groq:
    """
    Initialise the Groq client using GROQ_API_KEY from the environment.
    Raises a clear error if the key is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your key from console.groq.com."
        )
    return Groq(api_key=api_key)


_SCORING_SYSTEM_PROMPT = """
You are an expert technical recruiter. Your job is to evaluate a candidate's resume
against a job description and return a structured JSON score.

Rules:
- Be objective. Base your score purely on the resume content vs. the JD requirements.
- Do NOT penalise for irrelevant personal information.
- Score on a scale of 0 to 10 (float, one decimal place).
- Return ONLY valid JSON — no markdown, no explanation outside the JSON object.

Required output format:
{
  "score": <float 0-10>,
  "matched_skills": [<list of skills from resume that match JD>],
  "missing_skills": [<list of skills in JD not found in resume>],
  "experience_years": <estimated years of relevant experience as int, or null if unclear>,
  "education_fit": <"Exceeds" | "Meets" | "Below" | "Not Stated">,
  "strengths": "<2-3 sentence narrative of the candidate's key strengths for this role>",
  "gaps": "<1-2 sentence narrative of what is missing or weak>",
  "recommendation": <"Strong Hire" | "Hire" | "Maybe" | "No Hire">
}
""".strip()


def _build_user_prompt(jd_text: str, resume_text: str, candidate_name: str) -> str:
    """
    Construct the user-facing prompt for the LLM scorer.
    Truncates inputs if very long to stay within token limits.
    """
    # Rough truncation to ~4000 chars each to stay under context limits
    jd_snippet = jd_text[:4000].strip()
    resume_snippet = resume_text[:4000].strip()

    return (
        f"CANDIDATE: {candidate_name}\n\n"
        f"=== JOB DESCRIPTION ===\n{jd_snippet}\n\n"
        f"=== RESUME ===\n{resume_snippet}\n\n"
        "Evaluate this candidate against the JD and return the JSON score."
    )


def score_resume(
    jd_text: str,
    resume_text: str,
    candidate_name: str,
    client: Groq | None = None,
) -> dict:
    """
    Call the Groq LLM to score a single resume against the JD.

    Args:
        jd_text:        Full job description text.
        resume_text:    Full resume text.
        candidate_name: Filename or display name of the candidate.
        client:         Optional pre-built Groq client (reused for efficiency).

    Returns:
        A dict with keys: score, matched_skills, missing_skills,
        experience_years, education_fit, strengths, gaps, recommendation.
        Returns a default zeroed-out dict on failure.
    """
    if client is None:
        client = _get_client()

    if not resume_text.strip():
        logger.warning("Empty resume text for '%s' — assigning zero score.", candidate_name)
        return _default_score(candidate_name, reason="Empty resume — could not extract text.")

    user_prompt = _build_user_prompt(jd_text, resume_text, candidate_name)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,   # low temperature for consistent structured output
                max_tokens=600,
            )

            raw = response.choices[0].message.content.strip()
            result = _parse_json_response(raw, candidate_name)
            return result

        except json.JSONDecodeError as exc:
            logger.warning(
                "JSON parse error for '%s' (attempt %d/%d): %s",
                candidate_name, attempt, MAX_RETRIES, exc,
            )
        except Exception as exc:
            logger.warning(
                "API error for '%s' (attempt %d/%d): %s",
                candidate_name, attempt, MAX_RETRIES, exc,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    logger.error("All retries exhausted for '%s'. Using default zero score.", candidate_name)
    return _default_score(candidate_name, reason="LLM scoring failed after retries.")


def _parse_json_response(raw: str, candidate_name: str) -> dict:
    """
    Parse and validate the LLM's JSON response.
    Strips markdown code fences if the model adds them despite instructions.
    """
    # Strip optional markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    data = json.loads(raw)

    # Normalise score to [0, 1] for consistency with semantic score
    raw_score = float(data.get("score", 0))
    normalised_score = max(0.0, min(1.0, raw_score / 10.0))

    return {
        "llm_raw_score": raw_score,
        "llm_score": normalised_score,
        "matched_skills": data.get("matched_skills", []),
        "missing_skills": data.get("missing_skills", []),
        "experience_years": data.get("experience_years"),
        "education_fit": data.get("education_fit", "Not Stated"),
        "strengths": data.get("strengths", ""),
        "gaps": data.get("gaps", ""),
        "recommendation": data.get("recommendation", "Maybe"),
    }


def _default_score(candidate_name: str, reason: str = "") -> dict:
    """Return a zeroed-out score dict when scoring fails."""
    logger.debug("Default score assigned to '%s': %s", candidate_name, reason)
    return {
        "llm_raw_score": 0.0,
        "llm_score": 0.0,
        "matched_skills": [],
        "missing_skills": [],
        "experience_years": None,
        "education_fit": "Not Stated",
        "strengths": reason or "Unable to score.",
        "gaps": "N/A",
        "recommendation": "No Hire",
    }


def batch_score_resumes(
    jd_text: str,
    resumes: dict[str, str],
    delay_between_calls: float = 1.0,
) -> dict[str, dict]:
    """
    Score all resumes in sequence, reusing a single Groq client.

    Args:
        jd_text:               Full job description text.
        resumes:               Dict mapping filename → resume text.
        delay_between_calls:   Seconds to wait between API calls (rate limiting).

    Returns:
        Dict mapping filename → score dict.
    """
    client = _get_client()
    results: dict[str, dict] = {}

    for idx, (filename, text) in enumerate(resumes.items(), start=1):
        logger.info("LLM scoring %d/%d: %s", idx, len(resumes), filename)
        results[filename] = score_resume(jd_text, text, filename, client=client)

        # Small delay to stay within Groq free-tier rate limits
        if idx < len(resumes):
            time.sleep(delay_between_calls)

    return results
