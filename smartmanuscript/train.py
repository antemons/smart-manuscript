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
    flags.DEFINE_string(
        'profiling_steps', 'None',
        "List of steps to profile, e.g. [0, 10, 100] "
        "saves a file in timeline which can be opened in chrome "
        "via chrome://tracing/")
    flags.DEFINE_boolean(
        'share_param_first_layer',
        True,
        "Whether the first LSTM-layer shares parameters"
        "in forward and backward layer")
    flags.DEFINE_integer(
        'epoch_num', 1, "Number of epochs to train")
    flags.DEFINE_integer(
        'steps_per_checkpoint', 100, "")
    #flags.DEFINE_integer(
    #    'num_steps', 10000, "Number of batches to run")
    flags.DEFINE_float(
       'learning_rate', 0.003, "Learning rate for Optimizer")
    flags.DEFINE_boolean(
       'fine_tuning', True, r"reduce learning rate by a factor of 1/5 for last epoch")
    #flags.DEFINE_integer(
    #   'num_final_steps', 1000,
    #   "Number of batches to run before learning_rate_fine is used")
    flags.DEFINE_string(
       'data_path', "./records",
       "filepath to the tfrecords")
    flags.DEFINE_integer(
        'batch_size', 32, "Size of the training batch")

    return flags.FLAGS


def main():
    """ Train the network
    """
    FLAGS = read_flags()

    model = TrainingModel(
        lstm_sizes=json.loads(FLAGS.lstm_sizes),
        share_param_first_layer=FLAGS.share_param_first_layer)

    train_path_patterns = (4 * [os.path.join(FLAGS.data_path,"train/iam_line/*.tfrecord")] +
        [os.path.join(FLAGS.data_path, "train/ibm/*.tfrecord"),
         os.path.join(FLAGS.data_path,"train/iam_word/*.tfrecord"),
         os.path.join(FLAGS.data_path,"train/my_train/*.tfrecord")])

    test_path_patterns = {
        "ibm": os.path.join(FLAGS.data_path,"test/ibm/*.tfrecord"),
        "iam_line": os.path.join(FLAGS.data_path,"test/iam_line/*.tfrecord"),
        "iam_word": os.path.join(FLAGS.data_path,"test/iam_word/*.tfrecord"),
        "zen_test": os.path.join(FLAGS.data_path,"test/zen_test/*.tfrecord")}

    model.train(
        dataset_patterns=train_path_patterns,
        path=FLAGS.path,
        batch_size=FLAGS.batch_size,
        test_datasets_pattern=test_path_patterns,
        epoch_num=FLAGS.epoch_num,
        steps_per_checkpoint=FLAGS.steps_per_checkpoint,
        fine_tuning=FLAGS.fine_tuning,
        learning_rate=FLAGS.learning_rate,
        profiling_steps=json.loads(FLAGS.profiling_steps))

if __name__ == "__main__":
    main()
