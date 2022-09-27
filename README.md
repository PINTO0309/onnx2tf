# [WIP] onnx2tf
Self-Created Tools to convert ONNX files (NCHW) to TensorFlow format (NHWC). The purpose of this tool is to solve the massive Transpose extrapolation problem in [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) ([onnx-tf](https://pypi.org/project/onnx-tf/)).

# Key concept
- [ ] onnx-tensorflow is a very useful tool, but the performance of the generated TensorFlow models is significantly degraded due to the extrapolation of a large number of `Transpose` OPs before and after each OP during the format conversion from `NCHW` to `NHWC`. Therefore, I will make this tool myself as a derivative tool of [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) without extrapolating `Transpose`.
- [ ] Not only does it handle conversions of 4-dimensional inputs, such as `NCHW` to `NHWC`, but also the number of input dimensions in 3, 5, or even more dimensions. For example, `NCDHW` to `NDHWC`, etc. However, since 1-D, 2-D, 3-D and 6-D input may produce patterns that are mechanically difficult to convert, it should be possible to give parameters to externally modify the tool's behavior.
- [ ] Support conversion to TensorFlow saved model and TFLite (Float32/Float16).
- [ ] Does not support quantization to INT8. For quantization, use the official TensorFlow converter to convert from saved_model to your own.
- [ ] Files exceeding the Protocol Buffers file size limit of 2GB are not supported. Therefore, the external format is not supported at the initial stage of tool creation.
- [ ] If there are ONNX OPs that are not supported by TensorFlow, use [simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools) to replace them with harmless OPs in advance and then use this tool to convert them. In other words, you can convert any model with your efforts.
