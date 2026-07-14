"""Official ONNX node backend tests executed through nxrt."""

from onnx.backend.test import BackendTest

from nxrt_backend import NxrtBackend


backend_test = BackendTest(NxrtBackend, __name__)
backend_test.enable_report()

# The node-model group is self-contained in the ONNX package. Other groups
# include full models and converted framework cases that are outside this
# offline, single-operator coverage target.
globals().update(
    {
        name: case
        for name, case in backend_test.test_cases.items()
        if name == "OnnxBackendNodeModelTest"
    }
)
