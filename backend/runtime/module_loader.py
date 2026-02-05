"""Dynamic loading helpers for generated env modules."""

from __future__ import annotations

import importlib.util
import tempfile
import uuid
from pathlib import Path
from types import ModuleType


def load_module_from_code(env_code: str, prefix: str) -> tuple[ModuleType, Path]:
    """Load Python module from source code and return module + file path."""

    temp_dir = Path(tempfile.mkdtemp(prefix=f"zerorl_{prefix}_"))
    module_path = temp_dir / "env.py"
    module_path.write_text(env_code, encoding="utf-8")

    module_name = f"zerorl_runtime_{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create import spec for generated env")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path


def resolve_env_class(module: ModuleType) -> type:
    """Return first class ending with Env."""

    for name, obj in vars(module).items():
        if isinstance(obj, type) and name.endswith("Env"):
            return obj
    raise RuntimeError("No environment class ending with 'Env' found")
