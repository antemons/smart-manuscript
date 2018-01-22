#!/usr/bin/env python3

"""
    This file is part of Smart Manuscript.

    Smart Manuscript (transcript handwritten notes or inputs)
    Copyright (c) 2017 Daniel Vorberg

    Smart Manuscript is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Smart Manuscript is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Smart Manuscript.  If not, see <http://www.gnu.org/licenses/>.
"""

from tensorflow.python.platform.app import flags
import json
import glob
import os
import tensorflow as tf

from .model import TrainingModel
#from .encoder import encoder

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def read_flags():
    flags.DEFINE_string(
        'restore_from', "",
        "optional ckpt-file to restore-from")
    flags.DEFINE_string(
        'name', "vanilla", "Name of the network")
    flags.DEFINE_string(
        'lstm_sizes', "[120, 120]",  "List of LSTM-layer sizes (json)")
    flags.DEFINE_boolean(
        'share_param_first_layer',
        True,
        "Whether the first LSTM-layer shares parameters"
        "in forward and backward layer")
    flags.DEFINE_integer(
        'num_steps', 10000, "Number of batches to run")
    flags.DEFINE_float(
       'learning_rate', 0.003, "Learning rate for Optimizer")
    flags.DEFINE_float(
       'learning_rate_fine', 0.0006, "Learning rate for fine-tuning")
    flags.DEFINE_integer(
       'num_final_steps', 1000,
       "Number of batches to run before learning_rate_fine is used")
    flags.DEFINE_string(
       'data_path', "tmp",
       "filepath to the tfrecords")
    #flags.DEFINE_boolean(
    #    'batch_size', 50, "Size of the training batch")
    # flags.DEFINE_integer(
    #     'test_batch_size_lines', 10,
    #     "Batch size of the test batch(es)")

    return flags.FLAGS




# def _dataset_from_tfrecords(filenames):
#
# #      Dataset.list_files(...).shuffle(num_shards).
# # Use dataset.interleave(lambda filename: tf.data.TextLineDataset(filename), cycle_length=N) to mix together records from N different shards.
# # Use dataset.shuffle(B) to shuffle the resulting dataset. Setting B might require some experimentation, but you will p
#
#
#     if all(isinstance(s, str) for s in filenames):
#         filenames = tf.constant(filenames)
#     dataset = tf.data.TFRecordDataset(filenames)
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(
#         lambda inputs, label: (inputs, tf.shape(inputs)[0], label))
#     dataset = dataset.padded_batch(
#         batch_size=10,
#         padded_shapes=(tf.TensorShape([None, 15]),
#                        tf.TensorShape([]),
#                        tf.TensorShape([])))
#     return dataset




# def get_all_datasets(path):
#     filenames = {}
#     for subfolders in glob.glob(os.path.join(path, "*")):
#         filenames[os.path.basename(subfolders)] = glob.glob(
#             os.path.join(subfolders, "*.tfrecords"))
#     return filenames


def main():
    """ Train the network
    """
    FLAGS = read_flags()

    model = TrainingModel(
        lstm_sizes=json.loads(FLAGS.lstm_sizes),
        share_param_first_layer=FLAGS.share_param_first_layer)

    train_path_patterns = [
        #"records/train/ibm/*.tfrecords",
        "records/train/iam_line/*.tfrecords",
        "records/train/iam_word/*.tfrecords"]
        #"records/train/my_train/*.tfrecords"]

    test_path_patterns = {
        "ibm": "records/test/ibm/*.tfrecords",
        "iam_line": "records/test/iam_line/*.tfrecords",
        "ia_word": "records/test/iam_word/*.tfrecords",
        "zen_test": "records/test/zen_test/*.tfrecords"}

    model.train(
        dataset_patterns=train_path_patterns,
        path="FLAGS.name",
        test_datasets_pattern=test_path_patterns)

if __name__ == "__main__":
    main()
