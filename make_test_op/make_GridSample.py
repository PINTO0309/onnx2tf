import tensorflow as tf

def bilinear_sample_noloop(image, grid):
    """
    image
        [N, H, W, C]
    grid
        [N, grid_H, grid_W, 2]
    """
    Nt, H, W, C = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]
    xgrid, ygrid = tf.split(
        value=grid,
        num_or_size_splits=2,
        axis=-1,
    )
    mask = tf.cast(
        (xgrid >= 0) & (ygrid >= 0) & (xgrid < W - 1) & (ygrid < H - 1),
        dtype=tf.float32,
    )
    x0 = tf.math.floor(xgrid)
    x1 = x0 + 1
    y0 = tf.math.floor(ygrid)
    y1 = y0 + 1

    wa = tf.transpose(
        a=(x1 - xgrid) * (y1 - ygrid),
        perm=[3, 0, 1, 2],
    )
    wb = tf.transpose(
        a=(x1 - xgrid) * (ygrid - y0),
        perm=[3, 0, 1, 2],
    )
    wc = tf.transpose(
        a=(xgrid - x0) * (y1 - ygrid),
        perm=[3, 0, 1, 2],
    )
    wd = tf.transpose(
        a=(xgrid - x0) * (ygrid - y0),
        perm=[3, 0, 1, 2],
    )

    x0 = tf.cast(
        tf.reshape(
            tensor=(x0 * mask),
            shape=[Nt, grid_H, grid_W],
        ),
        dtype=tf.int64,
    )
    y0 = tf.cast(
        tf.reshape(
            tensor=(y0 * mask),
            shape=[Nt, grid_H, grid_W]
        ),
        dtype=tf.int64,
    )
    x1 = tf.cast(
        tf.reshape(
            tensor=(x1 * mask),
            shape=[Nt, grid_H, grid_W]
        ),
        dtype=tf.int64,
    )
    y1 = tf.cast(
        tf.reshape(
            tensor=(y1 * mask),
            shape=[Nt, grid_H, grid_W]
        ),
        dtype=tf.int64,
    )

    ind = tf.range(limit=Nt)
    ind = tf.reshape(tensor=ind, shape=[Nt, 1])
    ind = tf.tile(input=ind, multiples=[1, grid_H])
    ind = tf.reshape(tensor=ind, shape=[Nt, grid_H, 1])
    ind = tf.tile(input=ind, multiples=[1, 1, grid_W])
    ind = tf.cast(ind, dtype=tf.int64)

    image = tf.transpose(
        a=image,
        perm=[3,0,1,2],
    )
    output_tensor = \
        image[:, ind, y0, x0] * wa \
        + image[:, ind, y1, x0] * wb \
        + image[:, ind, y0, x1] * wc \
        + image[:, ind, y1, x1] * wd
    output_tensor = tf.transpose(
        a=output_tensor,
        perm=[1,2,3,0],
    )
    mask = tf.tile(
        input=mask,
        multiples=[1,1,1,C],
    )
    output_tensor = output_tensor * mask

    return output_tensor

