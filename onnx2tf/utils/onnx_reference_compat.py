from __future__ import annotations

from typing import Any

import numpy as np
import onnx


def create_reference_evaluator(
    onnx_graph: onnx.ModelProto,
) -> Any:
    """Create an ONNX ReferenceEvaluator with complete forward-LSTM outputs."""
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.ops.op_lstm import LSTM as ReferenceLSTM

    class LSTM(ReferenceLSTM):
        def _run(
            self,
            X: np.ndarray,
            W: np.ndarray,
            R: np.ndarray,
            B: np.ndarray | None = None,
            sequence_lens: np.ndarray | None = None,
            initial_h: np.ndarray | None = None,
            initial_c: np.ndarray | None = None,
            P: np.ndarray | None = None,
            activation_alpha: Any = None,
            activation_beta: Any = None,
            activations: Any = None,
            clip: Any = None,
            direction: Any = None,
            hidden_size: Any = None,
            input_forget: Any = None,
            layout: Any = None,
        ) -> tuple[np.ndarray, ...]:
            if int(self.n_outputs) <= 2:
                return super()._run(
                    X,
                    W,
                    R,
                    B,
                    sequence_lens,
                    initial_h,
                    initial_c,
                    P,
                    activation_alpha,
                    activation_beta,
                    activations,
                    clip,
                    direction,
                    hidden_size,
                    input_forget,
                    layout,
                )

            direction_value = direction or "forward"
            if isinstance(direction_value, bytes):
                direction_value = direction_value.decode("utf-8")
            if int(W.shape[0]) != 1 or str(direction_value).lower() != "forward":
                raise NotImplementedError(
                    "ReferenceEvaluator three-output LSTM fallback supports "
                    "one forward direction only."
                )

            activation_names = [] if activations is None else list(activations)
            activation_names = [
                (
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else str(value)
                ).lower()
                for value in activation_names
            ]
            if activation_names and activation_names != ["sigmoid", "tanh", "tanh"]:
                raise NotImplementedError(
                    "ReferenceEvaluator three-output LSTM fallback supports "
                    "default forward activations only."
                )
            if activation_alpha is not None or activation_beta is not None:
                raise NotImplementedError(
                    "ReferenceEvaluator three-output LSTM fallback does not "
                    "support activation alpha/beta overrides."
                )
            if clip is not None and float(clip) != 0.0:
                raise NotImplementedError(
                    "ReferenceEvaluator three-output LSTM fallback does not support clip."
                )
            if input_forget is not None and int(input_forget) != 0:
                raise NotImplementedError(
                    "ReferenceEvaluator three-output LSTM fallback does not support input_forget."
                )

            runtime_layout = int(layout if layout is not None else self.layout)
            x_runtime = np.asarray(X)
            if runtime_layout != 0:
                x_runtime = np.swapaxes(x_runtime, 0, 1)

            w_runtime = np.squeeze(np.asarray(W), axis=0)
            r_runtime = np.squeeze(np.asarray(R), axis=0)
            hidden_size_runtime = int(r_runtime.shape[-1])
            if hidden_size is not None and int(hidden_size) != hidden_size_runtime:
                raise ValueError(
                    "LSTM hidden_size does not match recurrent weights. "
                    f"attribute={hidden_size} weights={hidden_size_runtime}"
                )
            batch_size = int(x_runtime.shape[1])
            if sequence_lens is not None:
                sequence_lens_runtime = np.asarray(sequence_lens).reshape(-1)
                if not np.all(sequence_lens_runtime == int(x_runtime.shape[0])):
                    raise NotImplementedError(
                        "ReferenceEvaluator three-output LSTM fallback supports "
                        "only full-length sequences."
                    )

            b_runtime = (
                np.zeros(8 * hidden_size_runtime, dtype=x_runtime.dtype)
                if B is None
                else np.asarray(B)
            )
            if b_runtime.ndim > 1 and int(b_runtime.shape[0]) == 1:
                b_runtime = np.squeeze(b_runtime, axis=0)
            p_runtime = (
                np.zeros(3 * hidden_size_runtime, dtype=x_runtime.dtype)
                if P is None
                else np.asarray(P)
            )
            if p_runtime.ndim > 1 and int(p_runtime.shape[0]) == 1:
                p_runtime = np.squeeze(p_runtime, axis=0)

            h_runtime = (
                np.zeros((batch_size, hidden_size_runtime), dtype=x_runtime.dtype)
                if initial_h is None
                else np.asarray(initial_h)
            )
            c_runtime = (
                np.zeros((batch_size, hidden_size_runtime), dtype=x_runtime.dtype)
                if initial_c is None
                else np.asarray(initial_c)
            )
            state_direction_axis = 0 if runtime_layout == 0 else 1
            if (
                h_runtime.ndim > 2
                and int(h_runtime.shape[state_direction_axis]) == 1
            ):
                h_runtime = np.squeeze(h_runtime, axis=state_direction_axis)
            if (
                c_runtime.ndim > 2
                and int(c_runtime.shape[state_direction_axis]) == 1
            ):
                c_runtime = np.squeeze(c_runtime, axis=state_direction_axis)

            y_runtime = np.empty(
                (
                    int(x_runtime.shape[0]),
                    1,
                    batch_size,
                    hidden_size_runtime,
                ),
                dtype=x_runtime.dtype,
            )
            p_i, p_o, p_f = np.split(p_runtime, 3)
            for step_index, x_step in enumerate(x_runtime):
                gates = (
                    np.dot(x_step, np.transpose(w_runtime))
                    + np.dot(h_runtime, np.transpose(r_runtime))
                    + np.add(*np.split(b_runtime, 2))
                )
                i_gate, o_gate, f_gate, c_gate = np.split(gates, 4, axis=-1)
                i_gate = self.f(i_gate + p_i * c_runtime)
                f_gate = self.f(f_gate + p_f * c_runtime)
                c_gate = self.g(c_gate)
                c_runtime = f_gate * c_runtime + i_gate * c_gate
                o_gate = self.f(o_gate + p_o * c_runtime)
                h_runtime = o_gate * self.h(c_runtime)
                y_runtime[int(step_index), 0, :, :] = h_runtime

            y_h = h_runtime[np.newaxis, ...]
            y_c = c_runtime[np.newaxis, ...]
            if runtime_layout != 0:
                y_runtime = np.transpose(y_runtime, [2, 0, 1, 3])
                y_h = np.transpose(y_h, [1, 0, 2])
                y_c = np.transpose(y_c, [1, 0, 2])
            return (
                y_runtime.astype(X.dtype, copy=False),
                y_h.astype(X.dtype, copy=False),
                y_c.astype(X.dtype, copy=False),
            )

    return ReferenceEvaluator(onnx_graph, new_ops=[LSTM])
