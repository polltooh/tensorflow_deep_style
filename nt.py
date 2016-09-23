import tensorflow as tf
import numpy as np

def get_target_style(target_layer_value):
    features = np.reshape(target_layer_value, (-1, target_layer_value.shape[3]))
    gram = np.matmul(features.T,features)/ features.size 
    
    return gram

def get_curr_style(target_layer):
    batch_num, h, w, c = target_layer.get_shape().as_list()
    features = tf.reshape(target_layer, [-1, c])
    gram = tf.matmul(tf.transpose(features), features) / (h * w * c)
    return gram

def get_style_loss(target_gram, curr_gram):
    return tf.reduce_mean(tf.nn.l2_loss(target_gram - curr_gram)/ (4 * target_gram.size))


def get_content_loss(target_value, curr_value):
    return tf.reduce_mean(tf.nn.l2_loss(target_value -  curr_value));
    
