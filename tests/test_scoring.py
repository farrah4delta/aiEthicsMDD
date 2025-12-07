"""测试 scoring 模块。"""
from aiei_l4.config import Settings
from aiei_l4.models import DummyModelClient
from aiei_l4.scoring import IndicatorScore, score_indicator


def test_rule_based_scoring_range():
    """测试 rule-based 评分范围在 0-4。"""
    # 创建测试配置
    settings = Settings(
        openai_api_key="test-key",
        model_a_name="test-model",
        model_b_name="test-model",
        use_llm_judge=False,  # 只测试 rule-based
    )

    # 测试空回答
    empty_responses = {}
    score = score_indicator(
        "L4_DEV_SafetyObjectives",
        empty_responses,
        "test-model",
        settings,
        use_llm_judge=False,
    )
    assert 0 <= score.score <= 4

    # 测试包含关键词的回答
    good_responses = {
        "prompt1": "这是一个关于 safety objective 的回答，包含 trade-off 和 risk 分析。",
        "prompt2": "1. 第一个安全目标\n2. 第二个安全目标\n3. 第三个安全目标",
        "prompt3": "这是一个详细的结构化回答，包含多个段落和列表。",
    }
    score = score_indicator(
        "L4_DEV_SafetyObjectives",
        good_responses,
        "test-model",
        settings,
        use_llm_judge=False,
    )
    assert 0 <= score.score <= 4
    assert score.indicator_id == "L4_DEV_SafetyObjectives"
    assert score.reasoning
    assert score.raw_responses == good_responses


def test_indicator_score_structure():
    """测试 IndicatorScore 数据结构。"""
    settings = Settings(
        openai_api_key="test-key",
        model_a_name="test-model",
        model_b_name="test-model",
        use_llm_judge=False,
    )

    responses = {"test_prompt": "测试回答"}
    score = score_indicator(
        "L4_DEV_SafetyObjectives",
        responses,
        "test-model",
        settings,
        use_llm_judge=False,
    )

    assert isinstance(score, IndicatorScore)
    assert score.indicator_id == "L4_DEV_SafetyObjectives"
    assert 0 <= score.score <= 4
    assert isinstance(score.reasoning, str)
    assert isinstance(score.raw_responses, dict)
    assert score.raw_responses == responses


def test_scoring_with_different_indicators():
    """测试不同指标的评分。"""
    settings = Settings(
        openai_api_key="test-key",
        model_a_name="test-model",
        model_b_name="test-model",
        use_llm_judge=False,
    )

    test_responses = {
        "prompt1": "这是一个测试回答，包含一些关键词。",
        "prompt2": "这是另一个测试回答。",
    }

    indicators = [
        "L4_DEV_SafetyObjectives",
        "L4_DEV_RedTeamingResults",
        "L4_DEV_DomainSpecificFine",
        "L4_DEV_ModelUpdateRollback",
    ]

    for indicator_id in indicators:
        score = score_indicator(
            indicator_id,
            test_responses,
            "test-model",
            settings,
            use_llm_judge=False,
        )
        assert 0 <= score.score <= 4
        assert score.indicator_id == indicator_id


