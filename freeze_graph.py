import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def load_graph_def(model_path, sess=None):
    if os.path.isfile(model_path):
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        sess = sess if sess is not None else tf.get_default_session()
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)


def freeze_from_checkpoint(checkpoint_file, output_layer_name):

    model_folder = os.path.basename(checkpoint_file)
    output_graph = os.path.join(model_folder, checkpoint_file + '.pb')

    with tf.Session() as sess:

        load_graph_def(checkpoint_file)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        print("Exporting graph...")
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_layer_name.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('output_layer')
    args = parser.parse_args()
    freeze_from_checkpoint(checkpoint_file=args.model_path, output_layer_name=args.output_layer)