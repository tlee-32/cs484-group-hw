import tensorflow as tf

# Implementation by https://stackoverflow.com/a/46844409
def auc_roc(actual, predicted):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(predicted, actual)

    # Find all variables created for this metric
    metricVars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metricVars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # Update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value