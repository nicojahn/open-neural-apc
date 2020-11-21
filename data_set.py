import tensorflow as tf

def get_tf_data_set(x, l, a, batch_size=32, concat_length=5, training=True):
    n, _, h, w = x.shape

    # network parameter
    frame_shape = [h, w] # input shape
    y_shape = [l.shape[2]+a.shape[2]] # label shape (lower/upper bound * output shape + 1 accuracy mask)

    # defining dataset parameter
    data_set_size = n # number of samples in the data set (focused on the shuffle buffer_size)
    batch_padding_value = -1 # the padding value applied after concatenation

    # 2 random number generators which create with a 50:50 change either a -1 or a 1
    ds_r1 = tf.data.experimental.RandomDataset()
    ds_r2 = tf.data.experimental.RandomDataset()
    if training:
        ds_r1 = ds_r1.map(lambda x: (x%2)*2-1)
        ds_r2 = ds_r2.map(lambda x: (x%2)*2-1)
    else:
        ds_r1 = ds_r1.map(lambda x: 1)
        ds_r2 = ds_r2.map(lambda x: 1)
        
    # zipped into a tranformation dataset (if you need more random variables, just add them into here)
    ds_t = tf.data.Dataset.zip((ds_r1, ds_r2))

    # fetching data from ragged tensor and creating input dataset
    ds_x = tf.data.Dataset.from_tensor_slices(x)

    # fetching data from ragged tensor and creating the label dataset and the accuracy mask dataset
    ds_l = tf.data.Dataset.from_tensor_slices(l)
    ds_a = tf.data.Dataset.from_tensor_slices(a)
    # for convenience also zipped together
    ds_y = tf.data.Dataset.zip((ds_l, ds_a))

    # putting all the things together into 1 zipped dataset
    ds = tf.data.Dataset.zip((ds_x, ds_y, ds_t))

    # random shuffle the whole dataset and indefinitely repeat
    ds = ds.shuffle(batch_size*concat_length, reshuffle_each_iteration=False).batch(batch_size*concat_length, drop_remainder=True).unbatch()

    # batching the data before the concat op and windowed datasets
    ds = ds.window(batch_size*concat_length)

    def dataset_fn(ds_x, ds_y, ds_t):
        import numpy as np

        # the augmentation functions
        def random_flip_forward_backward(x, y, t):
            # using one random variable for reversing the sequence or not
            idx = t[0][0]
            (l, a) = y
            # which includes the time axis on x ...
            # as well as the label mask which is reverse in time and also the upper and lower bound are flipped
            return x[:,::idx,:,:], (l[:,::idx,::idx], a), t

        def random_flip_left_right(x, y, t):
            # using the other random variable here (both transformations are independent from each other)
            idx = t[1][0]
            (l, a) = y
            # flipping x from left to right
            # always concatenate the label and the accuracy mask in the last tranformation and drop t
            return x[:,:,:,::idx], tf.concat([l, a], axis=2)

        # combine the datasets again
        ds = tf.data.Dataset.zip((ds_x, ds_y, ds_t))

        # augment the data (mirror) on the time and spatial axis and combine the y masks
        ds = ds.batch(1).map(random_flip_forward_backward, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(random_flip_left_right, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()

        # batch the data in the proper format
        ds = ds.batch(batch_size).batch(1)

        # extract x and y component (t is not in the dataset anymore)  
        ds_x = ds.map(lambda x, y: x)
        ds_y = ds.map(lambda x, y: y)

        final_data_sets = []
        for ds, shape in [[ds_x, frame_shape], [ds_y, y_shape]]:
            # create initial ragged tensor for x
            initial_value_ragged = tf.ragged.constant(np.zeros((1, batch_size, 0, *shape)), \
                                                      ragged_rank=2, dtype=tf.float16, \
                                                      inner_shape=tuple(shape))
            '''
            TypeError: in user code:

            /home/shared/open-neural-apc/data_set.py:90 None  *
                ds = ds.flat_map(lambda ds_x, ds_y, ds_t: dataset_fn(ds_x, ds_y, ds_t))
            /home/shared/open-neural-apc/data_set.py:78 dataset_fn  *
                ds = ds.reduce(initial_value_ragged, lambda x, y: tf.concat([x,y], axis=2))
            /opt/conda/lib/python3.8/site-packages/tensorflow/python/data/ops/dataset_ops.py:2023 reduce  **
                if not issubclass(new_state_class, state_class):

            TypeError: issubclass() arg 1 must be a class
            '''
            # concatenating x
            ds = ds.reduce(initial_value_ragged, lambda x, y: tf.concat([x,y], axis=2))
            ds = tf.data.Dataset.from_tensor_slices(ds)
            # to_tensor() does pad the ragged tensor
            ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensors(x.to_tensor(batch_padding_value)))
            final_data_sets += [ds]

        # combine final output again (ready for training)
        return tf.data.Dataset.zip(tuple(final_data_sets))

    # apply to all windows the previous defined transformations
    ds = ds.flat_map(lambda ds_x, ds_y, ds_t: dataset_fn(ds_x, ds_y, ds_t))
    return ds.prefetch(tf.data.experimental.AUTOTUNE)