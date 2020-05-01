import tensorflow as tf
def fully_connect(x, weight_shape, name, biased=True, bias_init_value=0.0):
    with tf.compat.v1.variable_scope(name):
        weight = tf.compat.v1.get_variable("weight", weight_shape, initializer=tf.contrib.layers.xavier_initializer())
        o = tf.matmul(x, weight)
        if biased:
            b = tf.get_variable('bias', shape=[weight_shape[-1]], initializer=tf.constant_initializer(bias_init_value))
            o = o+b
    return o


def _dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, biased=False):
    num_x = x.shape[-1].value #C last
    with tf.compat.v1.variable_scope(name):
        w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o], initializer=tf.compat.v1.constant_initializer(0.0))
            o = tf.nn.bias_add(o, b)
    return o

def conv2d_transpose(x, filter_shape, output_shape, stride, name, padding='SAME', dilation=1, biased=True):
    with tf.compat.v1.variable_scope(name):
        w = tf.get_variable('weights', shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        o = tf.nn.conv2d_transpose(x, filters=w, output_shape=output_shape, 
            strides=stride, padding=padding, dilations=dilation)
        if biased:
            b = tf.get_variable('biases', shape=[filter_shape[-2]], initializer=tf.compat.v1.constant_initializer(0.0))  # output_channel input_channel
            o = tf.nn.bias_add(o, b)
    return o

def conv2d(x, filter_shape, stride, name, padding='SAME', dilation=1, biased=True):
    with tf.compat.v1.variable_scope(name):
        w = tf.get_variable('weights', shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        o = tf.nn.conv2d(x, filters=w, strides=stride, padding=padding, dilations=dilation)
        if biased:
            b = tf.get_variable('biases', shape=[filter_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
            o = tf.nn.bias_add(o, b)
    return o        

def deconv2d(x, filter_shape, output_size, name, padding='SAME', biased=True):
    with tf.compat.v1.variable_scope(name):
        x = tf.image.resize_images(x, size=output_size)
        w = tf.get_variable('weights', shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        o = tf.nn.conv2d(x, filters=w, strides=1, padding=padding)
        if biased:
            b = tf.get_variable('biases', shape=[filter_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
            o = tf.nn.bias_add(o, b)
    return o 

def InstanceNorm(x):
    return tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-5, trainable=False)

def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True,
             kernel_initializer=None):
    """Define conv for generator.
    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    x = tf.layers.conv2d(x,cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name,
        kernel_initializer=kernel_initializer)
    # We empirically found BN to help if not trained (works as regularizer)
    x = tf.layers.batch_normalization(x)
    x = activation(x)

    return x

def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.
    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    with tf.compat.v1.variable_scope(name):
        x = resize(x, func=tf.compat.v1.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x
    
def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.compat.v1.image.resize_bilinear, name='resize'):
    """
    This resize operation is used to scale the input according to some given
    scale and function.
    """

    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.compat.v1.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


def conv(inputs, name, shape, stride, padding='SAME', dilations=1 ,reuse=None, training=True, activation=tf.nn.leaky_relu,
         init_w=tf.contrib.layers.xavier_initializer_conv2d(), init_b=tf.constant_initializer(0.0)):
    with tf.compat.v1.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.contrib.framework.model_variable('weights', shape=shape, initializer=init_w, trainable=True)
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding, dilations=dilations)
        biases = tf.contrib.framework.model_variable('biases', shape=[shape[3]], initializer=init_b, trainable=True)
        conv = tf.nn.bias_add(conv, biases)
        if activation:
            conv = activation(conv)
        return conv

def deconv(inputs, size, name, shape, reuse=None, training=True, activation=tf.nn.leaky_relu):
    deconv = tf.image.resize_images( inputs, size=size )
    deconv = conv( deconv, name=name, shape=shape, stride=1, reuse=reuse, training=training, activation=activation )
    return deconv