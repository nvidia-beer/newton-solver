"""Renaming must preserve SolverBase and already-suffixed package names."""

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path


class TestGenerator(unittest.TestCase):
    def test_clone_tokens(self):
        root = Path(__file__).resolve().parents[1]
        for stem, class_name in (("custom", "SolverCustom"), ("custom-method", "SolverCustomMethod")):
            with self.subTest(method=stem), tempfile.TemporaryDirectory() as directory:
                self.check_clone(root, Path(directory), stem, class_name)

    def test_initialize_clone_in_place(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            self.copy_template(root, checkout)
            for relative in (".git/history", ".venv/installed.txt", ".cache/build.txt"):
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Keep newton-solver, newton_solver, Solver unchanged.\n")
            skill = checkout / ".claude/skills/paper-to-newton-solver/SKILL.md"
            skill.write_text(skill.read_text() + "\nImport newton_solver and Solver from newton-solver.\n")
            subprocess.run(
                [sys.executable, str(checkout / "tools/new_solver.py"), "--in-place", "--name", "newton-solver-custom"],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
            for relative in (".git/history", ".venv/installed.txt", ".cache/build.txt"):
                self.assertEqual(
                    (checkout / relative).read_text(), "Keep newton-solver, newton_solver, Solver unchanged.\n"
                )
            self.assertIn("Import newton_solver_custom and SolverCustom from newton-solver-custom.", skill.read_text())
            self.check_project(checkout, "custom", "SolverCustom")

    @staticmethod
    def copy_template(root, target):
        shutil.copytree(
            root,
            target,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                ".cache",
                ".ruff_cache",
                "__pycache__",
                "*.pyc",
                "results",
                "*.nsys-rep",
                "*.sqlite",
                "*.ncu-rep",
            ),
        )

    def test_profiling_copy_excludes_results_and_discovers_renamed_package(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "template"
            self.copy_template(root, source)
            inherited = (
                "nsys/results/old-run/capture.nsys-rep",
                "nsys/results/old-run/run.json",
                "nsys/manual.nsys-rep",
                "nsys/manual.sqlite",
                "nsys/manual.ncu-rep",
            )
            for relative in inherited:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Profiling data from the template, not the new solver.\n")
            target = directory / "generated"
            subprocess.run(
                [sys.executable, str(source / "tools/new_solver.py"), str(target), "--name", "newton-solver-custom"],
                check=True,
                capture_output=True,
                text=True,
            )
            for relative in inherited:
                self.assertFalse((target / relative).exists(), relative)
            self.assertFalse((target / "nsys/results").exists())
            for relative in ("nsys/profile_examples.py", "nsys/runner.py", "nsys/report.py"):
                self.assertTrue((target / relative).is_file(), relative)
            # A newly recorded preview must resolve to the renamed package and
            # use the same NVTX domain in its runner and capture command.
            media = target / "docs/media"
            media.mkdir()
            (media / "drop.mp4").touch()
            (media / "drop.gif").touch()
            inspect = textwrap.dedent("""\
                import json
                import runpy
                import sys
                from pathlib import Path
                root = Path(sys.argv[1])
                sys.path.insert(0, str(root / 'nsys'))
                profile = runpy.run_path(str(root / 'nsys/profile_examples.py'))
                package, examples = profile['video_examples']()
                print(json.dumps({'package': package, 'module': examples['drop']['module']}))
                """)
            result = subprocess.run(
                [sys.executable, "-I", "-c", inspect, str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(result.stdout),
                {"package": "newton_solver_custom", "module": "newton_solver_custom.examples.example_drop"},
            )
            dry_run = subprocess.run(
                [sys.executable, str(target / "nsys/profile_examples.py"), "drop", "--dry-run"],
                check=True,
                capture_output=True,
                text=True,
            )
            command = json.loads(dry_run.stdout)["command"]
            self.assertIn("--nvtx-capture=profile@newton-solver-custom", command)
            self.assertIn(str(target / "nsys/runner.py"), command)

    def check_clone(self, root, directory, stem, class_name):
        target = directory / f"newton-solver-{stem}"
        subprocess.run(
            [sys.executable, str(root / "tools/new_solver.py"), str(target), "--name", target.name],
            check=True,
            capture_output=True,
            text=True,
        )
        self.check_project(target, stem, class_name)

    def check_project(self, target, stem, class_name):
        directory = target.parent
        name = f"newton-solver-{stem}"
        metadata = tomllib.loads((target / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["name"], name)
        module = name.replace("-", "_")
        family = stem.replace("-", "_")
        package = target / "src" / module
        tree = ast.parse((package / "_src/solvers" / family / "solver.py").read_text())
        solver = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        self.assertEqual(solver.name, class_name)
        self.assertEqual(solver.bases[0].id, "SolverBase")
        self.assertFalse((package / "_src/solver.py").exists())
        self.assertFalse((package / "_src/solvers/template").exists())
        self.assertIn(name, (target / "README.md").read_text())
        self.assertNotIn("newton-solver-custom-custom", (target / "README.md").read_text())
        self.assertNotIn("tools/new_solver.py", (target / "README.md").read_text())
        self.assertFalse((target / "tools/new_solver.py").exists())
        self.assertFalse((target / "tests/test_generator.py").exists())
        self.assertTrue((target / "tools/build_readme.py").exists())
        self.assertTrue((target / "tools/record_examples.py").exists())
        for script in ("profile_examples.py", "runner.py", "report.py"):
            self.assertTrue((target / "nsys" / script).is_file())
        runner = ast.parse((target / "nsys/runner.py").read_text())
        domain = next(
            node.value
            for node in runner.body
            if isinstance(node, ast.Assign)
            and any(isinstance(item, ast.Name) and item.id == "DOMAIN" for item in node.targets)
        )
        self.assertEqual(ast.literal_eval(domain), name)
        self.assertIn("profiling", metadata["project"]["optional-dependencies"])
        self.assertFalse((target / "docs/media").exists())
        self.assertIn("Video not recorded yet", (target / "README.md").read_text())
        skill = target / ".agents/skills/paper-to-newton-solver"
        self.assertTrue(skill.is_symlink())
        self.assertEqual(skill.readlink(), Path("../../.claude/skills/paper-to-newton-solver"))
        self.assertTrue((skill / "SKILL.md").is_file())
        before = (target / "README.md").read_text()
        subprocess.run(
            [sys.executable, str(target / "tools/build_readme.py")],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(before, (target / "README.md").read_text())
        lock = tomllib.loads((target / "uv.lock").read_text())
        self.assertTrue(any(item["name"] == name for item in lock["package"]))
        category = package / "examples/research/inverse"
        category.mkdir(parents=True)
        for folder in (category, category.parent):
            (folder / "__init__.py").write_text("")
        extra = category / "example_calibration.py"
        extra.write_text(
            '"""A newly cloned calibration example."""\n'
            "HEADLESS = True\n"
            'raise RuntimeError("discovery must not import examples")\n'
            'if __name__ == "__main__":\n    pass\n'
        )
        dispatcher = package / "examples/__main__.py"
        listed = subprocess.run(
            [sys.executable, str(dispatcher), "--list"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("calibration", listed.stdout)
        self.assertIn("A newly cloned calibration example.", listed.stdout)
        selected = subprocess.run(
            [sys.executable, str(dispatcher), "--select", "calibration"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(selected.stdout, "calibration headless\n")
        self.assertEqual(selected.stderr, "")
        extra.write_text(
            '"""Run an added nested example after renaming."""\nif __name__ == "__main__":\n    print(__package__)\n'
        )
        dispatch_script = textwrap.dedent(f"""\
            import runpy
            import sys
            sys.path.insert(0, sys.argv[1])
            sys.argv = ['examples', 'calibration']
            runpy.run_module('{module}.examples', run_name='__main__')
            """)
        dispatched = subprocess.run(
            [sys.executable, "-I", "-c", dispatch_script, str(target / "src")],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(dispatched.stdout.strip(), f"{module}.examples.research.inverse")
        script = textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, sys.argv[1])
                import newton
                import numpy as np
                from {module} import {class_name}, PlaneCollider
                from {module}._src.solvers import {class_name} as family_solver
                assert {class_name} is family_solver
                assert {class_name}.__module__ == '{module}._src.solvers.{family}.solver'
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(1.0, 0.0, 0.0), mass=1.0)
                model = builder.finalize(device='cpu')
                before, after = model.state(), model.state()
                solver = {class_name}(model, collider=PlaneCollider())
                solver.step(before, after, None, None, 0.01)
                np.testing.assert_allclose(after.particle_q.numpy(), [[0.01, 0.0, 1.0]], atol=1e-7)
                """)
        subprocess.run(
            [sys.executable, "-I", "-c", script, str(target / "src")],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_in_place_collision_preserves_checkout(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ("src/newton_solver_custom", "src/newton_solver/_src/solvers/custom"):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as directory:
                checkout = Path(directory) / "checkout"
                self.copy_template(root, checkout)
                occupied = checkout / relative
                occupied.mkdir(parents=True)
                (occupied / "sentinel.txt").write_text("Existing implementation")
                before = (checkout / "pyproject.toml").read_bytes()
                result = subprocess.run(
                    [
                        sys.executable,
                        str(checkout / "tools/new_solver.py"),
                        "--in-place",
                        "--name",
                        "newton-solver-custom",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("destination already exists", result.stderr)
                self.assertEqual((checkout / "pyproject.toml").read_bytes(), before)
                self.assertEqual((occupied / "sentinel.txt").read_text(), "Existing implementation")
                self.assertTrue((checkout / "tools/new_solver.py").is_file())
                self.assertTrue((checkout / "src/newton_solver/_src/solvers/template/solver.py").is_file())

    def test_broken_readme_aborts_before_in_place_changes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            self.copy_template(root, checkout)
            (checkout / "docs/method.md").unlink()
            metadata = (checkout / "pyproject.toml").read_bytes()
            result = subprocess.run(
                [sys.executable, str(checkout / "tools/new_solver.py"), "--in-place", "--name", "newton-solver-custom"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("docs/method.md", result.stderr)
            self.assertEqual((checkout / "pyproject.toml").read_bytes(), metadata)
            self.assertTrue((checkout / "src/newton_solver/_src/solvers/template/solver.py").is_file())
            self.assertTrue((checkout / "tools/new_solver.py").is_file())
            self.assertFalse((checkout / "src/newton_solver_custom").exists())

    def test_destination_mode_is_unambiguous(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for options in ([], ["--in-place", str(Path(directory) / "extra")]):
                result = subprocess.run(
                    [sys.executable, str(root / "tools/new_solver.py"), *options, "--name", "newton-solver-custom"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("destination", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_family_name(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for name in ("newton-solver-3d", "newton-solver-class"):
                target = Path(directory) / name
                result = subprocess.run(
                    [sys.executable, str(root / "tools/new_solver.py"), str(target), "--name", name],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("valid Python family identifier", result.stderr)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
