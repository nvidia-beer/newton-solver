"""Recorded examples must honor their CLI options and per-step validation."""

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Recording requires the examples extra")
class TestRecording(unittest.TestCase):
    def test_terminal_iterations_reach_optimizer(self):
        from newton_solver.examples import example_diffsim_optimize as example

        hook = _tool("video_scenarios")
        result = {"history_m2": [1.0, 0.1], "initial_loss_m2": 1.0, "final_loss_m2": 0.1}
        with patch.object(example, "optimize_velocity", return_value=result) as optimize:
            images = list(
                hook.record_terminal("diffsim_optimize", ["--iterations", "1", "--device", "cpu"], 2, 320, 240)
            )
        optimize.assert_called_once_with(iterations=1, device="cpu")
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].shape, (240, 320, 3))

    def test_unknown_terminal_options_fail(self):
        hook = _tool("video_scenarios")
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as caught:
            next(hook.record_terminal("diffsim_optimize", ["--unknown-option"], 1, 320, 240))
        self.assertEqual(caught.exception.code, 2)

    def test_post_step_failure_stops_recording_and_closes_viewer(self):
        import newton.examples
        import newton.viewer

        recorder = _tool("record_examples")
        events = []

        class Viewer:
            def __init__(self, **kwargs):
                pass

            def close(self):
                events.append("close")

        def validate():
            events.append("post_step")
            raise AssertionError("bad physical state")

        def run_module(*args, **kwargs):
            example = SimpleNamespace(
                viewer=newton.viewer.ViewerGL(),
                step=lambda: events.append("step"),
                test_post_step=validate,
                test_final=lambda: events.append("final"),
                render=lambda: events.append("render"),
            )
            newton.examples.run(example, SimpleNamespace(test=True))

        with tempfile.TemporaryDirectory() as directory:
            options = SimpleNamespace(
                frames=2, fps=30, output_dir=Path(directory), arguments=["--test"], device="cpu", width=320, height=240
            )
            with (
                patch.object(recorder, "ROOT", Path(directory)),
                patch.object(
                    recorder, "project", return_value=({}, "newton_solver", {"drop": ("Drop", False, "drop")})
                ),
                patch.object(recorder.runpy, "run_module", side_effect=run_module),
                patch.object(newton.viewer, "ViewerGL", Viewer),
            ):
                original_argv, original_run = sys.argv, newton.examples.run
                with self.assertRaisesRegex(AssertionError, "bad physical state"):
                    recorder.record("drop", options)
                self.assertIs(sys.argv, original_argv)
                self.assertIs(newton.examples.run, original_run)
            self.assertFalse((Path(directory) / "drop.mp4").exists())
        self.assertEqual(events, ["step", "post_step", "close"])


if __name__ == "__main__":
    unittest.main()
