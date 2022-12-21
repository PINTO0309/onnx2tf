# Contribution Guide

## 1. Issues
- Right now, I am the sole maintainer of this repository. Therefore, my maintaining the code and cooperating with your research requires a great time cost and a strong motivation.

  Simply put, there are five factors that motivate me,

  1. `Purpose`: Personal development or Research or Product development. How much impact will solving this problem have on your efforts?
  2. `What`: What events are occurring? What error logs are being displayed?
  3. `How`: How did you attempt to solve the problem yourself?
  4. `Why`: Why do you need this problem solved?
  5. `Resources`: Material you have cited or referenced other than ONNX or JSON.

  In other words, I am mainly interested in how interesting your project is to me and why you are in trouble and how big an impact it will have on your project. The issue template has all the required fields for the minimum information you want me to provide to motivate me. Please describe the information as politely and gentlemanly as possible without omission.

- If your project's policy does not allow you to share ONNX files owned by your project, you may still politely submit your assignment and share only the ONNX files via email or other private means.
- Also, never post logs or error messages displayed in the console in an abbreviated form. Omission makes it difficult to understand the situation.
- I am not very impressed with issues that do not feel motivated. Therefore, if there is no response for more than 5 days after the last reply, the bot will automatically close the issue.

https://github.com/PINTO0309/onnx2tf/issues

## 2. Pull Request
- Basically any reasonable pull request is welcome. There are no strict rules.
- I would be very happy if you could share the ONNX file for testing if possible, as I would like to add the ONNX file for testing to the GitHub Actions for regression testing.
- When you issue a pull request, GitHub Actions will automatically run a regression test of the model transformation. Therefore, you can check the GitHub Actions execution log to see if your suggested modification affects previous code.
  - https://github.com/PINTO0309/onnx2tf/wiki/model_status
  - https://github.com/PINTO0309/onnx2tf/actions/workflows/test-models.yml
  ![image](https://user-images.githubusercontent.com/33194443/207194598-178d99e7-4ceb-44fc-a445-136bb513860d.png)
- Pull requests from engineers other than the repository owner will always fail to post the [model conversion status to the Wiki](https://github.com/PINTO0309/onnx2tf/wiki/model_status) due to lack of permissions to reference secrets, but there is no need to be concerned about this.

- To debug, tools must be installed as follows. If you do not want to destroy your environment, you can use [Docker](https://github.com/PINTO0309/onnx2tf/blob/main/Dockerfile).
  - HostPC install
    ```bash
    pip install -U pip \
    && pip install -U onnx \
    && pip install -U onnx-simplifier \
    && python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
    && pip install -U simple_onnx_processing_tools \
    && pip install tensorflow==2.10.0
    ```
  - Docker
    ```bash
    git clone https://github.com/PINTO0309/onnx2tf && cd onnx2tf
    docker build -t onnx2tf_develop .
    ```

https://github.com/PINTO0309/onnx2tf/pulls

## 3. Code Structure
### 3-1. Code hierarchy
```
.
├── CITATION.cff
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── LICENSE_onnx-tensorflow
├── README.md
├── demo.yml
├── onnx2tf
│   ├── __init__.py
│   ├── __main__.py
│   ├── onnx2tf.py             ... Main process of onnx2tf
│   ├── ops                    ... Various OP conversion processes compatible with ONNX OP Type name
│   │   ├── Abs.py
│   │   ├── Acos.py
│   │   ├── Acosh.py
:   :   :    :
│   │   ├── Xor.py
│   │   ├── _LSTM.py
│   │   ├── _Loop.py
│   │   └── __init__.py
│   └── utils                  ... common processing
│       ├── __init__.py
│       ├── colors.py
│       ├── common_functions.py
│       └── enums.py
└── setup.py
```
### 3-2. Code block structure
- `onnx2tf/ops/xxx.py`
  ```python
  @print_node_info
  @inverted_operation_enable_disable
  @get_replacement_parameter
  def make_node(
      *,
      graph_node: gs.Node,
      tf_layers_dict: dict,
      **kwargs: dict,
  ):

      # 1. ONNX parameter read section
      # 2. Preserving Graph Structure (Dict) section
      # 3. Param replacement section
      # 4. Generation of TF OP section
      # 5. Generation of Debug Info section
  ```
  #### 1. ONNX parameter read section
  1. `NCW` to `NWC`, `NCHW` to `NHWC`, `NCDHW` to `NDHWC`, etc... transposition determination

      It has no effect on most OPs, but is used to determine whether to transpose or not transpose the input of an OP after dimensional compression or dimensional expansion has occurred in `Reshape`. Determines if the ONNX output shape of the previous OP and the TensorFlow output shape are the same, returning `True` if they are different and `False` if they are the same. `True` indicates that the input data transposition operation must be performed; `False` indicates that the input data transposition operation need not be performed. `tf_layers_dict[before_op_name]['before_op_output_shape_trans']` stores the output shape comparison result of the previous OP as True/False. The shape determination process is performed in the `@inverted_operation_enable_disable` decorator.
      ```python
      before_op_output_shape_trans_1 = \
          tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
      before_op_output_shape_trans_2 = \
          tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
      before_op_output_shape_trans = \
          before_op_output_shape_trans_1 \
          and before_op_output_shape_trans_2
      ```

  2. Transposition of input values and conversion process to Numpy.ndarray

      Transposes input data based on `before_op_output_shape_trans` True/False; if `before_op_output_shape_trans=True`, transposition is performed; if `before_op_output_shape_trans=False`, no transposition is performed. In addition, if the parameter type of ONNX is `Initializer`, it is automatically converted to `Numpy.ndarray`.
      ```python
      graph_node_input_1 = get_constant_or_variable(
          graph_node.inputs[0],
          before_op_output_shape_trans,
      )
      graph_node_input_2 = get_constant_or_variable(
          graph_node.inputs[1],
          before_op_output_shape_trans,
      )
      ```

  3. Type annotation for debugging efficiency

      Although it does not affect the operation of the tool itself, type annotations are added to make the best use of Lint when debugging with IDEs such as VSCode.
      ```python
      boxes = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
          if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
      scores = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
          if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
      ```
  4. Read the attribute values that the OP has

      Obtain the attribute values that the OP of ONNX has. If no value exists, the default value follows the description in the official ONNX specification. [Operator Schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
      ```python
      center_point_box = graph_node.attrs.get('center_point_box', 0)
      ```
  #### 2. Preserving Graph Structure (Dict) section
  Generates metadata. It is needed to maintain the graph structure of the entire model.
  ```python
  graph_node_output: gs.Variable = graph_node.outputs[0]
  shape = graph_node_output.shape
  dtype = graph_node_output.dtype

  tf_layers_dict[graph_node_output.name] = {
      'optype': graph_node.op,
      'shape': shape,
      'dtype': dtype,
  }
  ```
  #### 3. Param replacement section
  `@get_replacement_parameter` Overwrites the ONNX value with the parameter for extrapolation read by the decorator. Reads a JSON format file written by the user and replaces ONNX parameters according to the described rules. [Parameter replacement](https://github.com/PINTO0309/onnx2tf#parameter-replacement). All information read from JSON is stored in `kwargs['op_rep_params']`. `param_target` can be either `'inputs'` or `'attributes'` or `'outputs'` or `op`.
  ```python
  boxes = replace_parameter(
      value_before_replacement=boxes,
      param_target='inputs',
      param_name=graph_node.inputs[0].name,
      **kwargs,
  )
  scores = replace_parameter(
      value_before_replacement=scores,
      param_target='inputs',
      param_name=graph_node.inputs[1].name,
      **kwargs,
  )
  ```
  #### 4. Generation of TF OP section
  Generate a TensorFlow OP and store it in `tf_layers_dict[graph_node_output.name]['tf_node']`. The entire graph structure is preserved in dict format.
  ```python
  tf_layers_dict[graph_node_output.name]['tf_node'] = \
      tf.math.abs(
          x=input_tensor,
          name=graph_node.name,
      )
  ```
  The structure of dict is assumed to be as follows when this is processed.
  ```python
  tf_layers_dict[graph_node_output.name] = {
      'optype': graph_node.op,
      'shape': shape,
      'dtype': dtype,
      'tf_node': (TensorFlow Op),
  }
  ```

  #### 5. Generation of Debug Info section
  Generates model conversion log information. `tf_op_type` can be set to TensorFlow OP as is or to any string. `tf_inputs` and `tf_outputs` can be set to any key value. The `@print_node_info` decorator automatically reads the information and outputs the conversion log to the console.
  ```python
  tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
      make_tf_node_info(
          node_info={
              'tf_op_type': tf.image.non_max_suppression,
              'tf_inputs': {
                  'boxes': tf_boxes,
                  'scores': tf_scores,
                  'max_output_boxes_per_class': max_output_boxes_per_class,
                  'iou_threshold': iou_threshold,
                  'score_threshold': score_threshold,
              },
              'tf_outputs': {
                  'output': tf_layers_dict[graph_node_output.name]['tf_node'],
              },
          }
      )
  ```
