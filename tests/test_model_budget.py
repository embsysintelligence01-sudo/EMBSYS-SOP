import unittest


class BudgetTests(unittest.TestCase):
    def test_thread_budget_preserves_session_inputs_outputs_and_providers(self):
        import numpy as np, onnx, onnxruntime as ort
        from onnx import helper, TensorProto
        from services.model_budget import apply_cpu_budget

        original = ort.InferenceSession
        try:
            apply_cpu_budget(2)
            graph = helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "identity",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
            )
            session = ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            value = np.array([[1.0, 2.0, 3.0]], np.float32)
            np.testing.assert_array_equal(session.run(None, {"x": value})[0], value)
            self.assertEqual(session.get_session_options().intra_op_num_threads, 2)
            self.assertEqual(
                session.get_session_options().get_session_config_entry(
                    "session.intra_op.allow_spinning"
                ),
                "0",
            )
        finally:
            ort.InferenceSession = original
