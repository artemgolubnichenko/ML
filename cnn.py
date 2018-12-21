import tensorflow as tf

import configuration


def conv_norm_relu(inputs, filters, kernel_size, training):
    with tf.variable_scope(None, default_name="cnr") as scope:
        сonv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, activation=tf.nn.relu)
        norm = tf.layers.batch_normalization(inputs=сonv, axis=-1, training=training)
        return norm


def build_model(features, training):
    conv_norm_relu1 = conv_norm_relu(features, 8, 5, training)
    pool1 = tf.layers.max_pooling2d(inputs=conv_norm_relu1, pool_size=[2, 2], strides=2)
    conv_norm_relu2 = conv_norm_relu(pool1, 16, 3, training)
    pool2 = tf.layers.max_pooling2d(inputs=conv_norm_relu2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 30 * 30 * 16])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.sigmoid)
    dense2 = tf.layers.dense(inputs=dense, units=28, activation=tf.nn.sigmoid)
    return dense2


def cnn_model_fn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True
    else:
        training = False

    logits = build_model(features, training)

    predicted_classes = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = binary_cross_entropy(logits, labels)

    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=configuration.LEARNING_RATE)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def binary_cross_entropy(logits, labels2, epsilon=1e-8, name='bce_loss'):
    # For brevity, let `x = logits`, `z = labels`.  The binary cross entropy loss is
    # loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    labels = tf.cast(labels2, tf.float32)
    return tf.reduce_mean(
        tf.reduce_sum(-(labels * tf.log(logits + epsilon) + (1. - labels) * tf.log(1. - logits + epsilon)), axis=1),
        name=name
    )
