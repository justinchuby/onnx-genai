from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from conformance import run_onnx_tests


def result(op: str, status: str, detail: str = "test detail") -> run_onnx_tests.Result:
    return run_onnx_tests.Result(op, "test", status, detail)


class ExitCodeTests(unittest.TestCase):
    def test_success_allows_passes_and_expected_unsupported(self) -> None:
        results = [
            result("Add", "PASS"),
            result("Abs", "PASS"),
            result(run_onnx_tests.SYNTHETIC_UNSUPPORTED_OP, "UNSUPPORTED"),
        ]
        expected_status = {
            "Add": "PASS",
            "Abs": "PASS",
            run_onnx_tests.SYNTHETIC_UNSUPPORTED_OP: "UNSUPPORTED",
        }

        self.assertEqual(
            run_onnx_tests.exit_code_for_results(results, expected_status), 0
        )
        self.assertEqual(
            run_onnx_tests.conformance_failures(results, expected_status), []
        )

    def test_mismatch_returns_non_zero(self) -> None:
        results = [result("Add", "MISMATCH", "values differ")]
        expected_status = {"Add": "PASS"}

        self.assertEqual(
            run_onnx_tests.exit_code_for_results(results, expected_status), 1
        )
        self.assertIn(
            "expected PASS, got MISMATCH",
            run_onnx_tests.conformance_failures(results, expected_status)[0],
        )

    def test_error_returns_non_zero(self) -> None:
        results = [result("Add", "ERROR", "runner crashed")]
        expected_status = {"Add": "PASS"}

        self.assertEqual(
            run_onnx_tests.exit_code_for_results(results, expected_status), 1
        )
        self.assertIn(
            "expected PASS, got ERROR",
            run_onnx_tests.conformance_failures(results, expected_status)[0],
        )

    def test_current_pass_becoming_unsupported_returns_non_zero(self) -> None:
        results = [
            result("Abs", "UNSUPPORTED", "no kernel"),
            result(run_onnx_tests.SYNTHETIC_UNSUPPORTED_OP, "UNSUPPORTED"),
        ]
        expected_status = {
            "Abs": "PASS",
            run_onnx_tests.SYNTHETIC_UNSUPPORTED_OP: "UNSUPPORTED",
        }

        self.assertEqual(
            run_onnx_tests.exit_code_for_results(results, expected_status), 1
        )
        self.assertIn(
            "Abs: expected PASS, got UNSUPPORTED",
            run_onnx_tests.conformance_failures(results, expected_status)[0],
        )

    def test_main_exits_non_zero_for_induced_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as work_dir:
            args = SimpleNamespace(
                onnx_tests=Path("."),
                runner=Path("synthetic-runner"),
                work_dir=Path(work_dir),
                json=None,
            )
            with (
                patch.object(run_onnx_tests, "EXPECTED_STATUS", {"Add": "PASS"}),
                patch.object(run_onnx_tests, "parse_args", return_value=args),
                patch.object(
                    run_onnx_tests, "generated_cases", return_value={"Add": object}
                ),
                patch.object(
                    run_onnx_tests,
                    "run_case",
                    return_value=result("Add", "MISMATCH", "forced mismatch"),
                ),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                with self.assertRaises(SystemExit) as caught:
                    raise SystemExit(run_onnx_tests.main())

        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
