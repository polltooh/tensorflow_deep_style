import tensorflow as tf
import cv2
import numpy as np

model_fn = 'tensorflow_inception_graph.pb'

graph = tf.Graph()
sess = tf.Session(graph = graph);

with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())



t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})


layers = [op.name for op in graph.get_operations() if op.type=='conv2d' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

print(sum(feature_nums))
