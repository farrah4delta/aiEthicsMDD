"""Evaluation pipeline module."""
import time
from dataclasses import dataclass
from typing import Dict, List

from aiei_l4.config import Settings
from aiei_l4.models import BaseModelClient, get_model_clients
from aiei_l4.prompts import PromptSet, get_prompt_sets
from aiei_l4.scoring import IndicatorScore, score_indicator


@dataclass
class ModelEvaluationResult:
    """Evaluation result for a single model."""

    model_name: str
    model_provider: str  # API provider: 'openai' or 'gemini'
    indicator_scores: List[IndicatorScore]


@dataclass
class ComparisonResult:
    """Comparison result for two models."""

    model_a: ModelEvaluationResult
    model_b: ModelEvaluationResult


def run_indicator_for_model(
    model_client: BaseModelClient,
    prompt_set: PromptSet,
    settings: Settings,
) -> IndicatorScore:
    """
    Run evaluation for a single indicator on a single model.

    Args:
        model_client: Model client
        prompt_set: Prompt set
        settings: Configuration object

    Returns:
        IndicatorScore object
    """
    start_time = time.perf_counter()
    responses: Dict[str, str] = {}

    # Iterate through all prompts
    for prompt in prompt_set.prompts:
        # Construct messages
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # Call model to generate response
        try:
            response = model_client.generate(messages)
            responses[prompt.id] = response
        except Exception as e:
            # If generation fails, record error message
            responses[prompt.id] = f"[ERROR: {e}]"

    # Perform scoring
    score = score_indicator(
        indicator_id=prompt_set.indicator_id,
        responses=responses,
        model_name=model_client.model_name,
        settings=settings,
        use_llm_judge=settings.use_llm_judge,
    )
    
    # Record duration
    duration = time.perf_counter() - start_time
    score.duration_seconds = duration
    
    return score


def run_full_evaluation(settings: Settings, progress_callback=None) -> ComparisonResult:
    """
    Run the complete evaluation pipeline.

    Args:
        settings: Configuration object
        progress_callback: Progress callback function that receives (current, total, message) parameters

    Returns:
        ComparisonResult object
    """
    total_start_time = time.perf_counter()
    
    # Create model clients
    client_a, client_b = get_model_clients(settings)

    # Get all prompt sets
    prompt_sets = get_prompt_sets()
    total_tasks = len(prompt_sets) * 2  # Two models
    current_task = 0

    # Evaluate model A
    if progress_callback:
        progress_callback(current_task, total_tasks, f"Starting evaluation for Model A: {settings.model_a_name}")
    else:
        print(f"Starting evaluation for Model A: {settings.model_a_name}")
    
    indicator_scores_a: List[IndicatorScore] = []
    for indicator_id, prompt_set in prompt_sets.items():
        current_task += 1
        if progress_callback:
            progress_callback(current_task, total_tasks, f"Evaluating Model A - {indicator_id}")
        else:
            print(f"  Evaluating indicator: {indicator_id}")
        
        score = run_indicator_for_model(client_a, prompt_set, settings)
        indicator_scores_a.append(score)
        
        if progress_callback:
            progress_callback(current_task, total_tasks, f"Model A - {indicator_id}: {score.score}/4 ({score.duration_seconds:.1f}s)")
        else:
            print(f"    Score: {score.score}/4 (Duration: {score.duration_seconds:.1f}s)")

    # Get actual model name and provider from client
    provider_a = (settings.model_a_provider or settings.api_provider).lower()
    model_result_a = ModelEvaluationResult(
        model_name=client_a.model_name,  # Use actual model name from client
        model_provider=provider_a,
        indicator_scores=indicator_scores_a,
    )

    # Evaluate model B
    if progress_callback:
        progress_callback(current_task, total_tasks, f"Starting evaluation for Model B: {settings.model_b_name}")
    else:
        print(f"\nStarting evaluation for Model B: {settings.model_b_name}")
    
    indicator_scores_b: List[IndicatorScore] = []
    for indicator_id, prompt_set in prompt_sets.items():
        current_task += 1
        if progress_callback:
            progress_callback(current_task, total_tasks, f"Evaluating Model B - {indicator_id}")
        else:
            print(f"  Evaluating indicator: {indicator_id}")
        
        score = run_indicator_for_model(client_b, prompt_set, settings)
        indicator_scores_b.append(score)
        
        if progress_callback:
            progress_callback(current_task, total_tasks, f"Model B - {indicator_id}: {score.score}/4 ({score.duration_seconds:.1f}s)")
        else:
            print(f"    Score: {score.score}/4 (Duration: {score.duration_seconds:.1f}s)")

    # Get actual model name and provider from client
    provider_b = (settings.model_b_provider or settings.api_provider).lower()
    model_result_b = ModelEvaluationResult(
        model_name=client_b.model_name,  # Use actual model name from client
        model_provider=provider_b,
        indicator_scores=indicator_scores_b,
    )

    total_duration = time.perf_counter() - total_start_time
    if progress_callback:
        progress_callback(total_tasks, total_tasks, f"Evaluation complete! Total duration: {total_duration:.1f}s")
    
    return ComparisonResult(model_a=model_result_a, model_b=model_result_b)

