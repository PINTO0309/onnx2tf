from onnx2tf.utils.flatbuffer_direct_op_error_report import (
    _sanitize_input_name_for_filename,
)


def test_sanitize_input_name_for_filename_replaces_path_separators() -> None:
    assert _sanitize_input_name_for_filename("gpu_0/data_0") == "gpu_0_data_0"
    assert _sanitize_input_name_for_filename(r"gpu_0\\data_0") == "gpu_0_data_0"


def test_sanitize_input_name_for_filename_handles_empty_like_name() -> None:
    assert _sanitize_input_name_for_filename("///") == "input"
