import importlib.util
from pathlib import Path


def load_module_from_path(module_name: str, file_path: Path):
    """Cleanly loads a Python file as a module without sys.path hacks."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"Could not load module {module_name} from {file_path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"Could not load module {module_name} from {file_path}"
    spec.loader.exec_module(module)
    return module
