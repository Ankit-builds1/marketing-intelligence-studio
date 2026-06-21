import pytest

gr = pytest.importorskip("gradio")

from app import gradio_app


def test_build_demo_returns_gradio_blocks():
    assert isinstance(gradio_app.build_demo(), gr.Blocks)
