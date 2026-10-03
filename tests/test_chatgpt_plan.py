import importlib.metadata


def test_plugin_is_installed():
    import llm_chatgpt_plan  # noqa: F401


def test_entry_point_registered():
    entry_points = importlib.metadata.entry_points(group="llm")
    assert any(
        ep.name == "chatgpt_plan" and ep.value == "llm_chatgpt_plan"
        for ep in entry_points
    )
