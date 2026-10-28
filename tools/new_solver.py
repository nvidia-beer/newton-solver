"""Initialize this clone in place, or copy it into a named independent solver project."""

import argparse
import keyword
import os
import re
import runpy
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, nargs="?", help="New project directory when copying the template")
    parser.add_argument("--in-place", action="store_true", help="Rename this clone, keeping its Git history")
    parser.add_argument("--name", required=True, help="Distribution name, e.g. newton-solver-custom")
    args = parser.parse_args()
    if not re.fullmatch(r"newton-solver-[a-z0-9]+(?:-[a-z0-9]+)*", args.name):
        parser.error("--name must be newton-solver- followed by lowercase words separated by hyphens")
    source = Path(__file__).resolve().parents[1]
    if args.in_place and args.destination is not None:
        parser.error("--in-place cannot be combined with a destination")
    if not args.in_place and args.destination is None:
        parser.error("provide a destination or use --in-place")
    target = source if args.in_place else args.destination.expanduser().resolve()
    if not args.in_place and (target.exists() or source == target or source in target.parents):
        parser.error("destination must be a new directory outside this template")
    module = args.name.replace("-", "_")
    stem = args.name.removeprefix("newton-solver-")
    family = stem.replace("-", "_")
    if not family.isidentifier() or keyword.iskeyword(family):
        parser.error("the method name must form a valid Python family identifier")
    class_name = "Solver" + "".join(word.title() for word in stem.split("-"))
    replacements = {"newton-solver": args.name, "newton_solver": module, "Solver": class_name}
    # Package names include hyphens. Do not turn the documentation's example
    # newton-solver-custom into newton-solver-custom-custom, or edit SolverBase.
    tokens = re.compile(r"(?<![\w-])newton-solver(?![\w-])|(?<!\w)newton_solver(?!\w)|\bSolver\b")
    # Check the template and rename destinations before changing any files.
    package = source / "src" / "newton_solver"
    solvers = package / "_src" / "solvers"
    metadata = tomllib.loads((source / "pyproject.toml").read_text())
    if (
        metadata["project"]["name"] != "newton-solver"
        or not (solvers / "template" / "solver.py").is_file()
        or not (solvers / "__init__.py").is_file()
    ):
        parser.error("this directory is no longer the uninitialized solver template")
    if (source / "src" / module).exists() or (source / "src" / module).is_symlink():
        parser.error(f"package destination already exists: src/{module}")
    if family != "template" and ((solvers / family).exists() or (solvers / family).is_symlink()):
        parser.error(f"solver family destination already exists: {solvers / family}")
    # A broken README source must not leave an in-place rename half finished.
    runpy.run_path(str(source / "tools/build_readme.py"))["make_readme"](source)
    exclude = shutil.ignore_patterns(
        ".git",
        ".venv",
        ".cache",
        ".ruff_cache",
        "__pycache__",
        "*.pyc",
        "dist",
        "build",
        "*.egg-info",
        "results",
        "*.nsys-rep",
        "*.sqlite",
        "*.ncu-rep",
        "imgui.ini",
        ".pytest_cache",
        ".coverage",
        ".coverage.*",
        "htmlcov",
        "coverage.xml",
        "new_solver.py",
        "test_generator.py",
    )

    def ignored(directory, names):
        names_to_skip = set(exclude(directory, names))
        if Path(directory) in (source / "docs", target / "docs"):
            # Starter recordings are not evidence for the newly named solver.
            names_to_skip.add("media")
        return names_to_skip

    if not args.in_place:
        shutil.copytree(source, target, ignore=ignored, symlinks=True)
    for directory, dirs, files in os.walk(target, followlinks=False):
        skip = ignored(directory, dirs + files)
        dirs[:] = [name for name in dirs if name not in skip]
        for name in files:
            path = Path(directory) / name
            if name in skip or path.is_symlink():
                continue
            try:
                content = path.read_text()
            except UnicodeDecodeError:
                continue
            renamed = tokens.sub(lambda match: replacements[match.group()], content)
            renamed = renamed.replace("_src/solvers/template", f"_src/solvers/{family}")
            if renamed != content:
                path.write_text(renamed)
    (target / "src" / "newton_solver").rename(target / "src" / module)
    solvers = target / "src" / module / "_src" / "solvers"
    if family != "template":
        (solvers / "template").rename(solvers / family)
    exports = solvers / "__init__.py"
    exports.write_text(exports.read_text().replace("from .template import", f"from .{family} import"))
    if args.in_place:
        media = target / "docs" / "media"
        if media.is_symlink() or media.is_file():
            media.unlink()
        elif media.exists():
            shutil.rmtree(media)
        (target / "tools/new_solver.py").unlink()
        (target / "tests/test_generator.py").unlink(missing_ok=True)
    subprocess.run(
        [sys.executable, str(target / "tools/build_readme.py")],
        cwd=target,
        check=True,
    )
    print(f"Initialized {target}\nImport: from {module} import {class_name}\nRun: cd {target} && uv sync --locked")


if __name__ == "__main__":
    main()
