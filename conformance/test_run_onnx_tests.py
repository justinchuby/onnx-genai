from __future__ import annotations

import unittest

from conformance import run_onnx_tests


def result(op: str, status: str, detail: str = "test detail") -> run_onnx_tests.Result:
    return run_onnx_tests.Result(op, "test", status, detail)


class ExitCodeTests(unittest.TestCase):
    def test_success_allows_passes_and_expected_unsupported(self) -> None:
        results = [
            result("Add", "PASS"),
            result("Abs", "PASS"),
            result("Conv", "UNSUPPORTED"),
        ]

        self.assertEqual(run_onnx_tests.exit_code_for_results(results), 0)
        self.assertEqual(run_onnx_tests.conformance_failures(results), [])

    def test_mismatch_returns_non_zero(self) -> None:
        results = [result("Add", "MISMATCH", "values differ")]

        self.assertEqual(run_onnx_tests.exit_code_for_results(results), 1)
        self.assertIn("Add: MISMATCH", run_onnx_tests.conformance_failures(results)[0])

    def test_error_returns_non_zero(self) -> None:
        results = [result("Add", "ERROR", "runner crashed")]

        self.assertEqual(run_onnx_tests.exit_code_for_results(results), 1)
        self.assertIn("Add: ERROR", run_onnx_tests.conformance_failures(results)[0])

    def test_supported_op_becoming_unsupported_returns_non_zero(self) -> None:
        results = [result("Add", "UNSUPPORTED", "no kernel")]

        self.assertEqual(run_onnx_tests.exit_code_for_results(results), 1)
        self.assertIn("expected PASS", run_onnx_tests.conformance_failures(results)[0])


if __name__ == "__main__":
    unittest.main()
