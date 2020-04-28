import os
import tensorflow.compat.v1 as tf


def standardize_name(name):
    return ''.join(w.capitalize() for w in name.split('_'))


def create_tf_session(use_gpu, gpu_frac=0.6, allow_gpu_growth=True, which_gpu=0):
    if use_gpu:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_frac,
            allow_growth=allow_gpu_growth)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            allow_soft_placement=True,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
    return tf.Session(config=config)


def test_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'while' for node in g.as_graph_def().node):
        print("{}({}) uses tf.while_loop.".format(
            f.__name__, ', '.join(map(str, args))))
    elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
        print("{}({}) uses tf.data.Dataset.reduce.".format(
            f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) gets unrolled.".format(
            f.__name__, ', '.join(map(str, args))))


def dump_string(string, filename):
    with open(filename, 'w+') as file:
        file.write(string)
