from typing import Any


def export_tflite_model_flatbuffer_direct(**kwargs: Any) -> None:
    output_folder_path = kwargs.get('output_folder_path', 'saved_model')
    output_file_name = kwargs.get('output_file_name', 'model')
    target_path = f'{output_folder_path}/{output_file_name}_float32.tflite'
    raise NotImplementedError(
        'tflite_backend="flatbuffer_direct" is not implemented yet. '
        f'Target file: {target_path}. '
        'Use tflite_backend="tf_converter" for now.'
    )
