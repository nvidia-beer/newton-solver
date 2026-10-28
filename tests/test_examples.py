"""Example discovery and selection use stdlib fixtures without solver imports."""

import json
import os
import pty
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class TestExamples(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        package = self.root / "fixture_solver"
        self.examples = package / "examples"
        self.examples.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (self.examples / "__init__.py").write_text("")
        project = Path(__file__).resolve().parents[1]
        self.dispatcher = self.examples / "__main__.py"
        shutil.copyfile(project / "src/newton_solver/examples/__main__.py", self.dispatcher)
        self.add_example("alpha", "Alpha geometry preview.")
        self.add_example("calibrate", "Fit material parameters.", headless="True")
        self.add_example("computed", "Computed metadata is not a literal.", headless="bool(1)")
        (self.examples / "example_bomb.py").write_text(
            '"""Discover this without importing its dependencies."""\n'
            'raise RuntimeError("LISTING_IMPORTED_AN_EXAMPLE")\n'
            'if __name__ == "__main__":\n    print("SHOULD_NOT_RUN")\n'
        )
        (self.examples / "example_helper.py").write_text(
            '"""HELPER_MUST_NOT_APPEAR"""\nraise RuntimeError("helper imported")\n'
        )
        (self.examples / "_private.py").write_text(
            '"""PRIVATE_MUST_NOT_APPEAR"""\nif __name__ == "__main__":\n    pass\n'
        )

    def add_example(self, name, description, headless=None, category=""):
        folder = self.examples
        for part in Path(category).parts:
            folder /= part
            folder.mkdir(exist_ok=True)
            (folder / "__init__.py").touch()
        metadata = "" if headless is None else f"HEADLESS = {headless}\n"
        source = f'"""{description}"""\n{metadata}' + textwrap.dedent(f"""\
            import json
            import sys

            if __name__ == "__main__":
                print(json.dumps({{"name": {name!r}, "args": sys.argv[1:]}}))
            """)
        path = folder / f"example_{name}.py"
        path.write_text(source)
        return path

    def run_cli(self, *args, as_file=False):
        command = (
            [sys.executable, str(self.dispatcher)] if as_file else [sys.executable, "-m", "fixture_solver.examples"]
        )
        return subprocess.run([*command, *args], cwd=self.root, input="", capture_output=True, text=True, timeout=5)

    def run_tty(self, answer, *args):
        master, slave = pty.openpty()
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "fixture_solver.examples", *args],
                cwd=self.root,
                stdin=slave,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            os.close(slave)
            slave = None
            try:
                os.write(master, answer.encode())
                stdout, stderr = process.communicate(timeout=5)
            except BaseException:
                process.kill()
                process.communicate()
                raise
            return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        finally:
            os.close(master)
            if slave is not None:
                os.close(slave)

    def test_lists_dynamic_examples_without_imports(self):
        self.add_example("zebra", "Newly added example appears automatically.")
        for as_file in (False, True):
            with self.subTest(as_file=as_file):
                result = self.run_cli("--list", as_file=as_file)
                self.assertEqual(result.returncode, 0, result.stderr)
                for text in ("alpha", "calibrate", "bomb", "zebra", "Alpha geometry preview.", "Newly added example"):
                    self.assertIn(text, result.stdout)
                self.assertNotIn("HELPER_MUST_NOT_APPEAR", result.stdout)
                self.assertNotIn("PRIVATE_MUST_NOT_APPEAR", result.stdout)
                self.assertNotIn("LISTING_IMPORTED_AN_EXAMPLE", result.stderr)

    def test_forwards_all_arguments_including_help(self):
        arguments = ["--help", "--device", "cpu", "space value", "--", "-n", "3"]
        result = self.run_cli("alpha", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"name": "alpha", "args": arguments})

    def test_nested_categories_discover_and_run_with_relative_imports(self):
        path = self.add_example("recover", "A nested inverse experiment.", headless="True", category="research/inverse")
        (path.parent / "support.py").write_text('NAME = "nested module resolved"\n')
        path.write_text(
            path.read_text()
            .replace("import json", "from .support import NAME\nimport json")
            .replace("'recover'", "NAME")
        )
        self.add_example("secret", "Hidden implementation.", category="_internal")
        self.add_example("invalid", "Not a Python package.", category="invalid-name")
        listed = self.run_cli("--list", as_file=True)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("recover", listed.stdout)
        self.assertIn("A nested inverse experiment.", listed.stdout)
        self.assertNotIn("secret", listed.stdout)
        self.assertNotIn("invalid", listed.stdout)
        selected = self.run_cli("--select", "recover", as_file=True)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(selected.stdout, "recover headless\n")
        arguments = ["--help", "--value", "space value"]
        executed = self.run_cli("recover", *arguments)
        self.assertEqual(executed.returncode, 0, executed.stderr)
        self.assertEqual(json.loads(executed.stdout), {"name": "nested module resolved", "args": arguments})

    def test_duplicate_commands_are_rejected_without_execution(self):
        self.add_example("alpha", "Conflicting nested command.", category="other")
        for args in (("--list",), ("alpha",), ("--select", "alpha")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Duplicate example name 'alpha'", result.stderr)
                self.assertIn("other.example_alpha", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_selector_uses_literal_metadata_and_only_prints_selection(self):
        for name, mode in (("alpha", "viewer"), ("calibrate", "headless"), ("computed", "viewer")):
            with self.subTest(name=name):
                result = self.run_cli("--select", name, as_file=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"{name} {mode}\n")
                self.assertEqual(result.stderr, "")

    def test_noninteractive_requires_selection(self):
        for args in ((), ("--select",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("alpha", result.stdout + result.stderr)
                self.assertNotIn('"name":', result.stdout)

    def test_invalid_name_does_not_execute(self):
        for args in (("missing",), ("--select", "missing"), ("../example_bomb",)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("SHOULD_NOT_RUN", result.stdout)
                self.assertNotIn("LISTING_IMPORTED_AN_EXAMPLE", result.stderr)
                self.assertNotIn('"name":', result.stdout)

    def test_interactive_default_number_and_name(self):
        for answer, name in (("\n", "alpha"), ("1\n", "alpha"), ("calibrate\n", "calibrate")):
            with self.subTest(answer=answer):
                result = self.run_tty(answer)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), {"name": name, "args": []})
                self.assertIn("alpha", result.stderr)

    def test_interactive_selector_and_quit(self):
        selected = self.run_tty("calibrate\n", "--select")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(selected.stdout, "calibrate headless\n")
        self.assertIn("alpha", selected.stderr)
        for args in ((), ("--select",)):
            with self.subTest(args=args):
                result = self.run_tty("q\n", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
