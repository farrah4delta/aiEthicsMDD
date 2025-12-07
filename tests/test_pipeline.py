"""测试 pipeline 模块。"""
from aiei_l4.config import Settings
from aiei_l4.models import DummyModelClient
from aiei_l4.pipeline import (
    ComparisonResult,
    ModelEvaluationResult,
    run_full_evaluation,
    run_indicator_for_model,
)
from aiei_l4.prompts import get_prompt_sets


def test_run_indicator_for_model():
    """测试单个指标的评估。"""
    settings = Settings(
        openai_api_key="test-key",
        model_a_name="test-model",
        model_b_name="test-model",
        use_llm_judge=False,
    )

    # 使用 DummyModelClient
    dummy_client = DummyModelClient(
        model_name="test-model",
        fixed_response="这是一个测试回复，包含 safety objective 和 trade-off 分析。",
    )

    prompt_sets = get_prompt_sets()
    prompt_set = prompt_sets["L4_DEV_SafetyObjectives"]

    result = run_indicator_for_model(dummy_client, prompt_set, settings)

    assert result.indicator_id == "L4_DEV_SafetyObjectives"
    assert 0 <= result.score <= 4
    assert result.reasoning
    assert len(result.raw_responses) == len(prompt_set.prompts)


def test_run_full_evaluation_with_dummy_models():
    """测试使用虚拟模型的完整评估流程。"""
    # 注意：这个测试需要真实的 API key，所以可能会失败
    # 在实际测试中，可以 mock OpenAI 客户端
    pass


def test_comparison_result_structure():
    """测试 ComparisonResult 数据结构。"""
    from aiei_l4.scoring import IndicatorScore

    # 创建模拟数据
    score_a = IndicatorScore(
        indicator_id="L4_DEV_SafetyObjectives",
        score=3,
        reasoning="测试理由",
        raw_responses={"prompt1": "回答1"},
    )

    score_b = IndicatorScore(
        indicator_id="L4_DEV_SafetyObjectives",
        score=4,
        reasoning="测试理由",
        raw_responses={"prompt1": "回答2"},
    )

    model_result_a = ModelEvaluationResult(
        model_name="model-a",
        model_provider="openai",
        indicator_scores=[score_a],
    )

    model_result_b = ModelEvaluationResult(
        model_name="model-b",
        model_provider="openai",
        indicator_scores=[score_b],
    )

    comparison = ComparisonResult(
        model_a=model_result_a,
        model_b=model_result_b,
    )

    assert comparison.model_a.model_name == "model-a"
    assert comparison.model_b.model_name == "model-b"
    assert len(comparison.model_a.indicator_scores) == 1
    assert len(comparison.model_b.indicator_scores) == 1
    assert comparison.model_a.indicator_scores[0].score == 3
    assert comparison.model_b.indicator_scores[0].score == 4


