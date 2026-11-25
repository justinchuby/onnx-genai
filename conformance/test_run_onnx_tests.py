from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
                cuda=False,
                smoke=False,
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


class RunCaseTests(unittest.TestCase):
    def test_current_reference_evaluator_reaches_output_comparison(self) -> None:
        import numpy as np

        model = run_onnx_tests.manual_model("Constant")

        def run_native(command: list[str], **_kwargs) -> SimpleNamespace:
            run_onnx_tests.write_tensor(
                Path(command[2]) / "output_0.nxrt",
                np.array([1.25, -2.5], dtype=np.float32),
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        native = Mock(side_effect=run_native)
        with (
            tempfile.TemporaryDirectory() as work_dir,
            patch.object(run_onnx_tests, "subprocess", SimpleNamespace(run=native)),
        ):
            actual = run_onnx_tests.run_case(
                "Constant", "test", model, Path("synthetic-runner"), Path(work_dir)
            )

        self.assertEqual(actual.status, "PASS", actual.detail)
        native.assert_called_once()

    def test_strict_cuda_requires_explicit_placement_and_distinguishes_errors(self) -> None:
        import numpy as np

        model = run_onnx_tests.smoke_model("ReluFloat32")
        for code, stdout, expected in [
            (0, "OK\t1 outputs on CUDA:0", "PASS"),
            (0, "OK\t1 outputs", "ERROR"),
            (2, "UNSUPPORTED_OP\tCUDA declined the node", "UNSUPPORTED"),
            (4, "UNSUPPORTED_FIXTURE\tNXRTCF01 cannot encode it", "UNSUPPORTED_FIXTURE"),
            (5, "SKIP\tgraph uses host dispatch", "SKIP"),
            (3, "ERROR\tCUDA kernel creation failed", "ERROR"),
            (3, "ERROR\tbuilt without the cuda feature", "ERROR"),
        ]:
            with self.subTest(code=code, stdout=stdout):
                def run_native(command: list[str], **_kwargs) -> SimpleNamespace:
                    self.assertEqual(command[1], "--cuda")
                    np.testing.assert_array_equal(
                        run_onnx_tests.read_tensor(Path(command[3]) / "input_0.nxrt"),
                        [-2.0, 0.0, 3.0],
                    )
                    run_onnx_tests.write_tensor(
                        Path(command[3]) / "output_0.nxrt",
                        np.array([0.0, 0.0, 3.0], dtype=np.float32),
                    )
                    return SimpleNamespace(returncode=code, stdout=stdout, stderr="")

                with (
                    tempfile.TemporaryDirectory() as work_dir,
                    patch.object(
                        run_onnx_tests, "subprocess", SimpleNamespace(run=run_native)
                    ),
                ):
                    actual = run_onnx_tests.run_case(
                        "ReluFloat32", "test", model, Path("runner"), Path(work_dir),
                        cuda=True, inputs=run_onnx_tests.smoke_inputs("ReluFloat32"),
                    )
                self.assertEqual(actual.status, expected, actual.detail)

    def test_unencodable_output_does_not_count_as_provider_decline(self) -> None:
        from onnx import TensorProto, helper

        value = helper.make_tensor("text", TensorProto.STRING, [1], ["hello"])
        graph = helper.make_graph(
            [helper.make_node("Constant", [], ["Y"], value=value)], "text", [],
            [helper.make_tensor_value_info("Y", TensorProto.STRING, [1])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
        )
        native = Mock()
        with (
            tempfile.TemporaryDirectory() as work_dir,
            patch.object(run_onnx_tests, "subprocess", SimpleNamespace(run=native)),
        ):
            actual = run_onnx_tests.run_case(
                "Constant", "test", model, Path("runner"), Path(work_dir), cuda=True
            )
        self.assertEqual(actual.status, "UNSUPPORTED_FIXTURE", actual.detail)
        self.assertNotEqual(run_onnx_tests.exit_code_for_results(
            [actual], {"Constant": "UNSUPPORTED"}
        ), 0)
        native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
