from __future__ import annotations

import importlib.util
from pathlib import Path



def test_inference_module_has_main() -> None:
    path = Path(__file__).resolve().parents[1] / "inference.py"
    spec = importlib.util.spec_from_file_location("return_desk_inference", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert callable(module.main)
