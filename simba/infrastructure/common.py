import os
import tensorflow.compat.v1 as tf


def standardize_name(name):
    return ''.join(w.capitalize() for w in name.split('_'))


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


def get_git_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def dump_string(string, filename):
    with open(filename, 'w+') as file:
        file.write(string)
