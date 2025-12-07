#!/usr/bin/env python3
"""Evaluation script entry point."""
import json
import sys
import time
from pathlib import Path

import typer

from aiei_l4.config import get_settings
from aiei_l4.pipeline import run_full_evaluation
from aiei_l4.report import to_json, to_markdown
from aiei_l4.utils import save_json, save_markdown

app = typer.Typer(help="AIEI L4 Indicator Evaluation Tool")


def format_time(seconds: float) -> str:
    """Format time display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def print_progress_bar(current: int, total: int, message: str, start_time: float):
    """Print progress bar."""
    if total == 0:
        return
    
    percentage = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    elapsed = time.perf_counter() - start_time
    if current > 0:
        avg_time_per_task = elapsed / current
        remaining_tasks = total - current
        estimated_remaining = avg_time_per_task * remaining_tasks
        time_info = f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(estimated_remaining)}"
    else:
        time_info = f"Elapsed: {format_time(elapsed)}"
    
    # Use \r and sys.stdout.write to update the same line
    sys.stdout.write(f"\r[{bar}] {percentage:.1f}% ({current}/{total}) | {message} | {time_info}")
    sys.stdout.flush()
    
    if current == total:
        print()  # New line after completion


@app.command()
def evaluate(
    model_a: str = typer.Option(
        None, "--model-a", "-a", help="Model A name (overrides environment variable)"
    ),
    model_b: str = typer.Option(
        None, "--model-b", "-b", help="Model B name (overrides environment variable)"
    ),
    output_json: str = typer.Option(
        "Reports/report.json", "--output-json", "-j", help="JSON report output path"
    ),
    output_md: str = typer.Option(
        "Reports/report.md", "--output-md", "-m", help="Markdown report output path"
    ),
    use_llm_judge: bool = typer.Option(
        True, "--use-llm-judge/--no-llm-judge", help="Whether to use LLM Judge for scoring"
    ),
):
    """
    Run the complete evaluation pipeline and generate reports.
    """
    # Read configuration
    settings = get_settings()

    # If command line provides parameters, override configuration
    if model_a:
        settings.model_a_name = model_a
    if model_b:
        settings.model_b_name = model_b
    settings.use_llm_judge = use_llm_judge

    typer.echo(f"Starting evaluation...")
    typer.echo(f"Model A: {settings.model_a_name}")
    typer.echo(f"Model B: {settings.model_b_name}")
    typer.echo(f"Using LLM Judge: {use_llm_judge}")
    typer.echo("")

    # Run evaluation (with progress bar)
    progress_start_time = time.perf_counter()
    
    def progress_callback(current: int, total: int, message: str):
        print_progress_bar(current, total, message, progress_start_time)
    
    try:
        result = run_full_evaluation(settings, progress_callback=progress_callback)
    except Exception as e:
        print()  # Ensure new line after progress bar
        typer.echo(f"Evaluation failed: {e}", err=True)
        raise typer.Exit(1)

    # Generate reports
    typer.echo("\nGenerating reports...")
    json_report = to_json(result, settings=settings)
    md_report = to_markdown(result, settings=settings)

    # Save reports
    save_json(json_report, output_json)
    save_markdown(md_report, output_md)

    typer.echo(f"✓ JSON report saved: {output_json}")
    typer.echo(f"✓ Markdown report saved: {output_md}")

    # Display summary
    summary = json_report["summary"]
    typer.echo("\n=== Evaluation Summary ===")
    typer.echo(f"Model A average score: {summary['average_score_a']}/4")
    typer.echo(f"Model B average score: {summary['average_score_b']}/4")
    typer.echo(f"Model A wins: {summary['wins_a']} indicators")
    typer.echo(f"Model B wins: {summary['wins_b']} indicators")
    typer.echo(f"Ties: {summary['ties']} indicators")
    typer.echo(f"Overall winner: {summary['overall_winner']}")


if __name__ == "__main__":
    app()


