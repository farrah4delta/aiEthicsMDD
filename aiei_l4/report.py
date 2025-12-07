"""Report generation module."""
from typing import Any, Dict

from aiei_l4.pipeline import ComparisonResult, IndicatorScore, ModelEvaluationResult


def to_json(comparison: ComparisonResult, settings=None) -> Dict[str, Any]:
    """
    Convert comparison result to JSON format.

    Args:
        comparison: Comparison result

    Returns:
        JSON dictionary
    """
    def indicator_score_to_dict(score: IndicatorScore) -> Dict[str, Any]:
        return {
            "indicator_id": score.indicator_id,
            "score": score.score,
            "reasoning": score.reasoning,
            "raw_responses": score.raw_responses,
            "duration_seconds": round(score.duration_seconds, 2),
        }

    def model_result_to_dict(result: ModelEvaluationResult) -> Dict[str, Any]:
        return {
            "model_name": result.model_name,
            "model_provider": result.model_provider,
            "indicator_scores": [
                indicator_score_to_dict(score) for score in result.indicator_scores
            ],
        }

    # 计算总耗时
    total_duration = sum(
        score.duration_seconds 
        for score in comparison.model_a.indicator_scores + comparison.model_b.indicator_scores
    )
    
    metadata = {
        "total_duration_seconds": round(total_duration, 2),
        "average_per_indicator_seconds": round(
            total_duration / (len(comparison.model_a.indicator_scores) * 2), 2
        ) if comparison.model_a.indicator_scores else 0,
    }
    
    # Add judge model info if available
    if settings:
        metadata["judge_model"] = settings.judge_model or "N/A"
        metadata["judge_model_provider"] = (settings.judge_model_provider or settings.api_provider) or "N/A"
        metadata["use_llm_judge"] = settings.use_llm_judge
        metadata["temperature"] = settings.temperature
    
    return {
        "model_a": model_result_to_dict(comparison.model_a),
        "model_b": model_result_to_dict(comparison.model_b),
        "summary": _generate_summary(comparison),
        "evaluation_metadata": metadata,
    }


def to_markdown(comparison: ComparisonResult, settings=None) -> str:
    """
    Convert comparison result to Markdown format.

    Args:
        comparison: Comparison result

    Returns:
        Markdown string
    """
    # Calculate total duration
    total_duration = sum(
        score.duration_seconds 
        for score in comparison.model_a.indicator_scores + comparison.model_b.indicator_scores
    )
    avg_duration = (
        total_duration / (len(comparison.model_a.indicator_scores) * 2)
        if comparison.model_a.indicator_scores
        else 0
    )
    
    def format_duration(seconds: float) -> str:
        """Format time display in English."""
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
    
    lines = [
        "# AIEI L4 Prompt-Based Evaluation Report",
        "",
        "## Models",
        "",
        f"- **Model A:** {comparison.model_a.model_name} ({comparison.model_a.model_provider.upper()})",
        f"- **Model B:** {comparison.model_b.model_name} ({comparison.model_b.model_provider.upper()})",
        "",
    ]
    
    # Add judge model info if available
    if settings:
        judge_model = settings.judge_model or "N/A"
        judge_provider = (settings.judge_model_provider or settings.api_provider) or "N/A"
        lines.extend([
            "## Evaluation Configuration",
            "",
            f"- **Judge Model:** {judge_model} ({judge_provider.upper()})" if settings.use_llm_judge else "- **Judge Model:** Not used (Rule-based only)",
            f"- **Temperature:** {settings.temperature}",
            f"- **Use LLM Judge:** {settings.use_llm_judge}",
            "",
        ])
    
    lines.extend([
        "## Evaluation Statistics",
        "",
        f"- **Total Duration:** {format_duration(total_duration)}",
        f"- **Average per Indicator:** {format_duration(avg_duration)}",
        "",
        "## Summary Table",
        "",
        "| Indicator | Model A Score | Model B Score | Duration (A) | Duration (B) | Winner |",
        "|-----------|---------------|---------------|--------------|---------------|--------|",
    ])

    # Generate summary table
    for score_a, score_b in zip(
        comparison.model_a.indicator_scores, comparison.model_b.indicator_scores
    ):
        assert score_a.indicator_id == score_b.indicator_id
        indicator_id = score_a.indicator_id
        score_a_val = score_a.score
        score_b_val = score_b.score
        duration_a = format_duration(score_a.duration_seconds)
        duration_b = format_duration(score_b.duration_seconds)

        if score_a_val > score_b_val:
            winner = "A"
        elif score_b_val > score_a_val:
            winner = "B"
        else:
            winner = "Tie"

        lines.append(
            f"| {indicator_id} | {score_a_val} | {score_b_val} | {duration_a} | {duration_b} | {winner} |"
        )

    lines.extend(["", "## Per-Indicator Details", ""])

    # Generate detailed information for each indicator
    for score_a, score_b in zip(
        comparison.model_a.indicator_scores, comparison.model_b.indicator_scores
    ):
        assert score_a.indicator_id == score_b.indicator_id
        indicator_id = score_a.indicator_id

        lines.extend([
            f"### {indicator_id}",
            "",
            f"- **Model A score:** {score_a.score}/4 (Duration: {format_duration(score_a.duration_seconds)})",
            f"- **Model B score:** {score_b.score}/4 (Duration: {format_duration(score_b.duration_seconds)})",
            "",
            "**Model A Reasoning:**",
            f"{score_a.reasoning}",
            "",
            "**Model B Reasoning:**",
            f"{score_b.reasoning}",
            "",
        ])

        # Add key evidence snippets
        lines.append("**Selected Evidence Snippets:**")
        lines.append("")

        # Extract snippets from Model A's responses
        if score_a.raw_responses:
            lines.append("*Model A:*")
            for prompt_id, response in list(score_a.raw_responses.items())[:2]:
                snippet = response[:200] + "..." if len(response) > 200 else response
                lines.append(f"- {prompt_id}: {snippet}")
            lines.append("")

        # Extract snippets from Model B's responses
        if score_b.raw_responses:
            lines.append("*Model B:*")
            for prompt_id, response in list(score_b.raw_responses.items())[:2]:
                snippet = response[:200] + "..." if len(response) > 200 else response
                lines.append(f"- {prompt_id}: {snippet}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _generate_summary(comparison: ComparisonResult) -> Dict[str, Any]:
    """Generate summary information."""
    total_a = sum(score.score for score in comparison.model_a.indicator_scores)
    total_b = sum(score.score for score in comparison.model_b.indicator_scores)
    avg_a = total_a / len(comparison.model_a.indicator_scores) if comparison.model_a.indicator_scores else 0
    avg_b = total_b / len(comparison.model_b.indicator_scores) if comparison.model_b.indicator_scores else 0

    wins_a = sum(
        1
        for score_a, score_b in zip(
            comparison.model_a.indicator_scores, comparison.model_b.indicator_scores
        )
        if score_a.score > score_b.score
    )
    wins_b = sum(
        1
        for score_a, score_b in zip(
            comparison.model_a.indicator_scores, comparison.model_b.indicator_scores
        )
        if score_b.score > score_a.score
    )
    ties = len(comparison.model_a.indicator_scores) - wins_a - wins_b

    return {
        "total_score_a": total_a,
        "total_score_b": total_b,
        "average_score_a": round(avg_a, 2),
        "average_score_b": round(avg_b, 2),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "overall_winner": "A" if avg_a > avg_b else "B" if avg_b > avg_a else "Tie",
    }

