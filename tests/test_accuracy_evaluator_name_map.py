from onnx2tf.tflite_builder.accuracy_evaluator import _build_tflite_detail_map


def _detail(name: str) -> dict:
    return {
        "name": name,
        "index": 0,
        "dtype": "float32",
    }


def test_build_tflite_detail_map_absorbs_arbitrary_signature_prefix() -> None:
    onnx_names = [
        "affine_src",
        "affine_dst",
        "aff_homo_img",
    ]
    # Keep order intentionally different from ONNX to ensure mapping is not
    # dependent on positional fallback.
    tflite_details = [
        _detail("sig_v2_main_aff_homo_img:0"),
        _detail("sig_v2_main_affine_src:0"),
        _detail("sig_v2_main_affine_dst:0"),
    ]

    mapped = _build_tflite_detail_map(
        onnx_names=onnx_names,
        tflite_details=tflite_details,
    )

    assert str(mapped["affine_src"]["name"]) == "sig_v2_main_affine_src:0"
    assert str(mapped["affine_dst"]["name"]) == "sig_v2_main_affine_dst:0"
    assert str(mapped["aff_homo_img"]["name"]) == "sig_v2_main_aff_homo_img:0"


def test_build_tflite_detail_map_prefers_longest_suffix_match() -> None:
    onnx_names = [
        "a",
        "b_a",
    ]
    tflite_details = [
        _detail("custom_signature_b_a:0"),
    ]

    mapped = _build_tflite_detail_map(
        onnx_names=onnx_names,
        tflite_details=tflite_details,
    )

    # Longest suffix ("b_a") should be selected for aliasing, and shorter
    # suffix ("a") should fall back positionally.
    assert str(mapped["b_a"]["name"]) == "custom_signature_b_a:0"
