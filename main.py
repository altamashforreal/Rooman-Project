"""
main.py — CLI entrypoint for the Resume Screening Agent

Usage:
    python main.py --jd data/job_description.txt --resumes data/resumes/ --output outputs/

The agent runs the full pipeline:
  1. Parse all resumes in the given folder
  2. Compute semantic similarity (sentence-transformers)
  3. Score each resume with the Groq LLM
  4. Combine scores and output a ranked shortlist

Results are printed to the terminal (rich table) and saved to:
  outputs/ranked_results.json
  outputs/ranked_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

from agent.parser import load_all_resumes, extract_text
from agent.ranker import rank_candidates

# Load .env before anything else so GROQ_API_KEY is available
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

console = Console()

# Colour mapping for recommendation labels
RECOMMENDATION_COLOURS = {
    "Strong Hire": "bold green",
    "Hire": "green",
    "Maybe": "yellow",
    "No Hire": "red",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume Screening Agent — rank resumes against a job description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --jd data/job_description.txt --resumes data/resumes/\n"
            "  python main.py --jd data/job_description.txt --resumes data/resumes/ --output outputs/\n"
        ),
    )
    parser.add_argument(
        "--jd",
        required=True,
        metavar="PATH",
        help="Path to the job description file (.txt, .pdf, or .docx).",
    )
    parser.add_argument(
        "--resumes",
        required=True,
        metavar="FOLDER",
        help="Path to the folder containing resume files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/",
        metavar="FOLDER",
        help="Folder to write ranked_results.json and .csv (default: outputs/).",
    )
    return parser.parse_args()


def print_banner() -> None:
    console.print()
    console.rule("[bold cyan]RESUME SCREENING AGENT[/bold cyan]")
    console.print(
        "[dim]Two-stage pipeline: semantic embeddings + Groq LLM reasoning[/dim]\n",
        justify="center",
    )


def print_results_table(ranked: list[dict]) -> None:
    """Render a rich table of ranked candidates to the terminal."""
    table = Table(
        title="Ranked Candidates",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )

    table.add_column("Rank", style="bold", width=5, justify="center")
    table.add_column("Candidate", min_width=22)
    table.add_column("Final Score", justify="center", width=12)
    table.add_column("Semantic", justify="center", width=10)
    table.add_column("LLM Score", justify="center", width=10)
    table.add_column("Exp (yrs)", justify="center", width=10)
    table.add_column("Recommendation", width=15)

    for c in ranked:
        rec = c.get("recommendation", "Maybe")
        rec_colour = RECOMMENDATION_COLOURS.get(rec, "white")
        exp = str(c.get("experience_years")) if c.get("experience_years") is not None else "—"

        table.add_row(
            str(c["rank"]),
            c["candidate"],
            f"[bold]{c['final_score']:.3f}[/bold]",
            f"{c['semantic_score']:.3f}",
            f"{c['llm_score']:.3f}",
            exp,
            f"[{rec_colour}]{rec}[/{rec_colour}]",
        )

    console.print(table)


def print_top_candidate_detail(candidate: dict) -> None:
    """Print a detailed breakdown of the top-ranked candidate."""
    console.print(f"\n[bold cyan]-- Top Candidate Detail: {candidate['candidate']} --[/bold cyan]")

    console.print(f"[bold]Matched Skills:[/bold] {', '.join(candidate['matched_skills']) or 'None identified'}")
    console.print(f"[bold]Missing Skills:[/bold] {', '.join(candidate['missing_skills']) or 'None'}")
    console.print(f"[bold]Education Fit:[/bold] {candidate['education_fit']}")
    console.print(f"\n[bold]Strengths:[/bold]\n  {candidate['strengths']}")
    console.print(f"\n[bold]Gaps:[/bold]\n  {candidate['gaps']}")


def main() -> None:
    args = parse_args()
    print_banner()

    # ── Validate inputs ───────────────────────────────────────────────────────
    jd_path = Path(args.jd)
    resume_folder = Path(args.resumes)

    if not jd_path.is_file():
        console.print(f"[bold red]Error:[/bold red] JD file not found: {jd_path}")
        sys.exit(1)

    if not resume_folder.is_dir():
        console.print(f"[bold red]Error:[/bold red] Resume folder not found: {resume_folder}")
        sys.exit(1)

    # ── Load JD ───────────────────────────────────────────────────────────────
    console.print(f"[cyan]Loading job description from:[/cyan] {jd_path}")
    jd_text = extract_text(str(jd_path))
    if not jd_text:
        console.print("[bold red]Error:[/bold red] Could not extract text from the JD file.")
        sys.exit(1)

    # ── Load resumes ──────────────────────────────────────────────────────────
    console.print(f"[cyan]Loading resumes from:[/cyan] {resume_folder}\n")
    try:
        resumes = load_all_resumes(str(resume_folder))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    console.print(f"[green]Found {len(resumes)} resume(s).[/green] Starting pipeline ...\n")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    # -- Run pipeline ----------------------------------------------------------
    ranked = rank_candidates(jd_text, resumes, output_dir=args.output)

    # -- Display results -------------------------------------------------------
    print_results_table(ranked)
    print_top_candidate_detail(ranked[0])

    console.print(f"\n[bold green]>> Results saved to:[/bold green] {args.output}")
    console.print(f"  - {args.output}ranked_results.json")
    console.print(f"  - {args.output}ranked_results.csv\n")


if __name__ == "__main__":
    main()
