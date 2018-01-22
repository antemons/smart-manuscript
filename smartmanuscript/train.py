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

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def read_flags():
    flags.DEFINE_string(
        'restore_from', "",
        "optional ckpt-file to restore-from")
    flags.DEFINE_string(
        'path', "vanilla", "Name of the network")
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


def main():
    """ Train the network
    """
    FLAGS = read_flags()

    model = TrainingModel(
        lstm_sizes=json.loads(FLAGS.lstm_sizes),
        share_param_first_layer=FLAGS.share_param_first_layer)

    train_path_patterns = [
        "records/train/ibm/*.tfrecord",
        "records/train/iam_line/*.tfrecord",
        "records/train/iam_word/*.tfrecord",
        "records/train/my_train/*.tfrecord"]

    test_path_patterns = {
        "ibm": "records/test/ibm/*.tfrecord",
        "iam_line": "records/test/iam_line/*.tfrecord",
        "ia_word": "records/test/iam_word/*.tfrecord",
        "zen_test": "records/test/zen_test/*.tfrecord"}

    model.train(
        dataset_patterns=train_path_patterns,
        path=FLAGS.path,
        test_datasets_pattern=test_path_patterns)

if __name__ == "__main__":
    main()
