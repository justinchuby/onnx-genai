"""Run with: python -m unittest discover -s tests -p test_python_loader.py."""

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
# The unit tests exercise Python discovery; installed-wheel smoke uses the real extension.
extension = types.ModuleType("onnx_genai._onnx_genai")
extension.Engine = object()
extension.GenerateResult = object()
extension.__version__ = "unit-test"
with patch.dict(sys.modules, {extension.__name__: extension}):
    import onnx_genai as launcher


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(importlib.invalidate_caches)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        standard_paths = [
            entry
            for entry in sys.path
            if not {"site-packages", "dist-packages"}.intersection(Path(entry).parts)
        ]
        self.path = patch.object(sys, "path", [str(self.root), *standard_paths])
        self.path.start()
        self.addCleanup(self.path.stop)
        self.addCleanup(launcher._DLL_DIRECTORY_HANDLES.clear)
        for name in ("onnxruntime", "onnxruntime_ep_mlx"):
            self.assertNotIn(name, sys.modules)

    def package(self, name):
        package = self.root / name
        package.mkdir()
        (package / "__init__.py").write_text(
            "raise AssertionError('optional package was imported')\n"
        )
        importlib.invalidate_caches()
        return package

    def test_native_public_exports_without_reference_install(self):
        self.assertIs(launcher.Engine, extension.Engine)
        self.assertIs(launcher.GenerateResult, extension.GenerateResult)
        self.assertEqual(launcher.__version__, "unit-test")
        launcher._configure_optional_libraries()
        self.assertNotIn("ONNX_GENAI_ORT_LIB_DIR", os.environ)
        self.assertNotIn("onnxruntime", sys.modules)

    def test_discovers_reference_without_importing_or_loading_it(self):
        capi = self.package("onnxruntime") / "capi"
        capi.mkdir()
        launcher._configure_optional_libraries()
        self.assertEqual(os.environ["ONNX_GENAI_ORT_LIB_DIR"], str(capi))
        self.assertNotIn("onnxruntime", sys.modules)

    def test_incomplete_reference_package_does_not_block_native(self):
        self.package("onnxruntime")
        launcher._configure_optional_libraries()
        self.assertNotIn("ONNX_GENAI_ORT_LIB_DIR", os.environ)

    def test_explicit_reference_hints_are_preserved(self):
        self.package("onnxruntime")
        for name in ("ONNX_GENAI_ORT_LIB", "ONNX_GENAI_ORT_LIB_DIR"):
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: "/explicit/runtime"}, clear=True),
            ):
                launcher._configure_optional_libraries()
                self.assertEqual(os.environ[name], "/explicit/runtime")
                self.assertEqual(len(os.environ), 1)

    def test_mlx_discovery_does_not_import_plugin(self):
        library = self.package("onnxruntime_ep_mlx") / "libonnxruntime_mlx_ep.dylib"
        library.touch()
        with patch.object(sys, "platform", "darwin"):
            launcher._configure_optional_libraries()
            self.assertEqual(os.environ["ONNX_GENAI_MLX_EP_LIBRARY"], str(library))
            with patch.dict(
                os.environ, {"ONNX_GENAI_MLX_EP_LIBRARY": "/explicit/plugin"}
            ):
                launcher._configure_optional_libraries()
                self.assertEqual(
                    os.environ["ONNX_GENAI_MLX_EP_LIBRARY"], "/explicit/plugin"
                )
        self.assertNotIn("onnxruntime_ep_mlx", sys.modules)

    def test_windows_explicit_reference_file_keeps_its_dependency_directory(self):
        library = self.root / "reference" / "onnxruntime.dll"
        library.parent.mkdir()
        library.touch()
        handle = object()
        with (
            patch.dict(os.environ, {"ONNX_GENAI_ORT_LIB": str(library)}),
            patch.object(sys, "platform", "win32"),
            patch.object(
                os, "add_dll_directory", create=True, return_value=handle
            ) as add,
        ):
            launcher._configure_optional_libraries()
        add.assert_called_once_with(str(library.parent))
        self.assertEqual(launcher._DLL_DIRECTORY_HANDLES, {str(library.parent): handle})
        self.assertNotIn("ONNX_GENAI_ORT_LIB_DIR", os.environ)

    def test_windows_retains_dependency_directory_handle(self):
        capi = self.package("onnxruntime") / "capi"
        capi.mkdir()
        handle = object()
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(
                os, "add_dll_directory", create=True, return_value=handle
            ) as add,
        ):
            launcher._configure_optional_libraries()
            launcher._configure_optional_libraries()
        add.assert_called_once_with(str(capi))
        self.assertEqual(launcher._DLL_DIRECTORY_HANDLES, {str(capi): handle})


if __name__ == "__main__":
    unittest.main()
