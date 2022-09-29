def convert_axis(
    *,
    axis: int,
    tensor_rank: int,
) -> int:
    """Convert axis from NCHW to NHWC or NCDHW to NDHWC. axis for rank numbers other than 4D and 5D do not convert.

    Parameters
    ----------
    axis: int
        Axis value to be replaced

    tensor_rank: int
        Number of ranks of ex-tensors specified by axis

    Returns
    ----------
    converted_axis: int
        Converted axis
    """
    # Convert a negative number of axis to a positive number
    converted_axis = axis if axis >= 0 else axis + tensor_rank

    # 4D and 5D axis conversion table
    convertion_table_4d = [0,3,1,2]
    convertion_table_5d = [0,4,1,2,3]

    if tensor_rank == 4:
        # NCHW -> NHWC
        converted_axis = convertion_table_4d[axis]

    elif tensor_rank == 5:
        # NCDHW -> NDHWC
        converted_axis = convertion_table_5d[axis]

    else:
        return converted_axis
