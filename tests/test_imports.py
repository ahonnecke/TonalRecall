"""
Test harness to verify all Python files can be imported without syntax errors.
"""

import importlib
import os
import sys
from pathlib import Path


def import_file(module_name, file_path):
    """Attempt to import a Python file and return any import errors."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return None
    except Exception as e:
        return str(e)


def find_python_files(directory):
    """Find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        # Skip hidden directories (like .git, __pycache__)
        if any(part.startswith(".") for part in Path(root).parts):
            continue

        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                full_path = os.path.join(root, file)
                # Convert to module path
                rel_path = os.path.relpath(full_path, directory)
                module_path = rel_path.replace("/", ".").replace("\\", ".")[
                    :-3
                ]  # Remove .py
                python_files.append((module_path, full_path))
    return python_files


def main():
    """Main test function to check all Python files for import errors."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_files = find_python_files(root_dir)

    print(f"Found {len(python_files)} Python files to test...\n")

    # Add the root directory to Python path for imports
    sys.path.insert(0, root_dir)

    errors = {}

    for module_path, file_path in python_files:
        print(f"Testing import of {module_path}... ", end="")
        error = import_file(module_path, file_path)
        if error:
            print("❌ FAILED")
            errors[module_path] = error
        else:
            print("✅ PASSED")

    print("\nTest Results:")
    print("-" * 80)

    if errors:
        print(f"Found {len(errors)} import errors:")
        for module, error in errors.items():
            print(f"\n{module}:")
            print(f"  {error}")
        return 1
    else:
        print("✅ All files imported successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
