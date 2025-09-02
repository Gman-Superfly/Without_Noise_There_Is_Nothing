from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import List


def load_plugins_from(folder: str = "optimizers") -> List[ModuleType]:
    assert isinstance(folder, str) and folder, "folder must be non-empty string"
    if folder not in sys.path:
        sys.path.insert(0, os.getcwd())  # ensure CWD in path for top-level package
    pkg_init = os.path.join(folder, "__init__.py")
    assert os.path.exists(pkg_init), f"Missing optimizer package: {folder}"

    modules: List[ModuleType] = []
    for entry in os.listdir(folder):
        if not entry.endswith(".py") or entry == "__init__.py":
            continue
        mod_name = f"{folder}.{entry[:-3]}"
        mod = importlib.import_module(mod_name)
        modules.append(mod)
    return modules


