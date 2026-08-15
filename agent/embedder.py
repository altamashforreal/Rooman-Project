"""
embedder.py — Semantic similarity via sentence-transformers

Uses the all-MiniLM-L6-v2 model to generate dense vector embeddings
for text, then computes cosine similarity between a JD and a resume.

This gives Stage 1 of our two-stage scoring pipeline — an objective,
keyword-and-concept overlap score that doesn't require an LLM call.

Model choice rationale:
  - all-MiniLM-L6-v2 is ~90 MB, fast on CPU, and scores well on
    semantic textual similarity benchmarks (STSB ~68 Spearman).
  - It's free, local, and reproducible — no API key, no rate limits. test
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Module-level singleton — load once, reuse across all calls.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformers model.
    Downloads on first call (~90 MB); cached locally afterwards.
    """
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model '%s' ...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Model loaded.")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Convert a text string into a 384-dimensional embedding vector.

    Args:
        text: Input string (JD or resume text).

    Returns:
        1-D numpy array of shape (384,).
    """
    model = _get_model()
    # encode() returns shape (384,) for a single string
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding


def compute_similarity(jd_text: str, resume_text: str) -> float:
    """
    Compute the cosine similarity between a job description and a resume.

    We embed both as full documents (truncated to 512 tokens by the model
    internally) and return a scalar in [0, 1].

    Args:
        jd_text:     Full text of the job description.
        resume_text: Full text of a single resume.

    Returns:
        Cosine similarity score as a float in [0.0, 1.0].
        Returns 0.0 if either input is empty.
    """
    if not jd_text.strip() or not resume_text.strip():
        logger.warning("Empty input passed to compute_similarity — returning 0.0.")
        return 0.0

    jd_vec = embed_text(jd_text).reshape(1, -1)
    resume_vec = embed_text(resume_text).reshape(1, -1)

    # cosine_similarity returns a 2-D array; extract the scalar
    score = float(cosine_similarity(jd_vec, resume_vec)[0][0])

    # Clamp to [0, 1] in case of floating-point edge cases
    return max(0.0, min(1.0, score))


def batch_similarity(jd_text: str, resumes: dict[str, str]) -> dict[str, float]:
    """
    Efficiently compute cosine similarity for a batch of resumes.

    Encodes all resumes in one forward pass for speed.

    Args:
        jd_text:  Full text of the job description.
        resumes:  Dict mapping filename → resume text.

    Returns:
        Dict mapping filename → similarity score [0, 1].
    """
    model = _get_model()

    filenames = list(resumes.keys())
    texts = list(resumes.values())

    logger.info("Embedding JD and %d resume(s) ...", len(texts))

    # Encode all at once for efficiency
    jd_vec = model.encode(jd_text, convert_to_numpy=True, show_progress_bar=False).reshape(1, -1)
    resume_vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # cosine_similarity(A, B) where A is (1, d) and B is (n, d) → shape (1, n)
    scores = cosine_similarity(jd_vec, resume_vecs)[0]

    return {
        filename: float(max(0.0, min(1.0, score)))
        for filename, score in zip(filenames, scores)
    }
