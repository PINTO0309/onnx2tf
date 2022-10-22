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
│   ├── ops                    ... Various OP conversion processes compatible with ONNX OP name
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

      It has no effect on most OPs, but is used to determine whether to transpose or not transpose the input of an OP after dimensional compression or dimensional expansion has occurred in `Reshape`. Determines if the ONNX output shape of the previous OP and the TensorFlow output shape are the same, returning `True` if they are different and `False` if they are the same. `True` indicates that the input data transposition operation must be performed; `False` indicates that the input data transposition operation need not be performed.
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
  3. Type annotation for debugging efficiency
  4. Read the attribute values that the OP has
  #### 2. Preserving Graph Structure (Dict) section
  #### 3. Param replacement section
  #### 4. Generation of TF OP section
  #### 5. Generation of Debug Info section
