"""测试 prompts 模块。"""
from aiei_l4.prompts import PromptSet, get_prompt_sets


def test_get_prompt_sets_returns_four_indicators():
    """测试 get_prompt_sets 返回 4 个指标。"""
    prompt_sets = get_prompt_sets()
    assert len(prompt_sets) == 4
    assert "L4_DEV_SafetyObjectives" in prompt_sets
    assert "L4_DEV_RedTeamingResults" in prompt_sets
    assert "L4_DEV_DomainSpecificFine" in prompt_sets
    assert "L4_DEV_ModelUpdateRollback" in prompt_sets


def test_each_indicator_has_at_least_two_prompts():
    """测试每个指标至少有 2 个 prompts。"""
    prompt_sets = get_prompt_sets()
    for indicator_id, prompt_set in prompt_sets.items():
        assert isinstance(prompt_set, PromptSet)
        assert len(prompt_set.prompts) >= 2, f"{indicator_id} 的 prompts 数量不足"
        assert prompt_set.indicator_id == indicator_id


def test_prompt_structure():
    """测试 Prompt 结构完整性。"""
    prompt_sets = get_prompt_sets()
    for prompt_set in prompt_sets.values():
        for prompt in prompt_set.prompts:
            assert prompt.id
            assert prompt.description
            assert prompt.system
            assert prompt.user



