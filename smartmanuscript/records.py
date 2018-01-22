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

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

import numpy as np
from random import choice, sample
import pylab as plt
from tensorflow.python.platform.app import flags
import glob
import os
import pickle
import tensorflow as tf

from . import corpus_iam
from . import corpus_ibm
from .corpus import Corpus, Corpora, TranscriptedStrokes
from .handwritten_vector_graphic import load as load_pdf
from .writing import strokes_to_features, plot_features, InkPage
from .preprocessing import preprocessed_corpus

def read_flags():
    flags.DEFINE_string(
        "show_record", "",
        "if a path is provided, it shows this tfrecord")
    flags.DEFINE_boolean(
        "load_cache", False,
        "load all strokes from the pkl file from previous run")
    flags.DEFINE_string(
        "path", "tmp", "folder to save the records")
    flags.DEFINE_integer(
        "max_files", 10**10,
        "the max number of files to read")
    flags.DEFINE_string("my_handwriting_train_path", "data/my_handwriting/train/",
                        "path folder with own examples")
    flags.DEFINE_string("my_handwriting_test_path", "data/my_handwriting/test/",
                        "path folder with own examples")
    return flags.FLAGS


def from_pdf(filename):
    """ Read strokes from a pdf/svg file and its correspoding transcription.

    The transcription must be in the file with same name und .txt suffix

    Args:
        filename (str): path to the pdf or svg file
    """
    transcription_filename = os.path.splitext(filename)[0] + ".txt"
    strokes, page_size = load_pdf(filename)
    page = InkPage(strokes, page_size)
    with open(transcription_filename) as f:
        transcriptions = [l.replace("\n", "") for l in f.readlines()]
    assert len(page.lines) == len(transcriptions), \
        filename + " %i != %i" % (len(page.lines), len(transcriptions))

    return Corpus(TranscriptedStrokes(transcription, line.strokes)
                  for transcription, line in zip(transcriptions, page.lines))


def read_pdf_folder(path, max_files=None):
    """ Load all strokes from pdf/svg and text from txt files of a folder.

    Read all pdf and svg files of a folder, extract its strokes and
    read its corresponding transcription from the txt-file of a folder

    Args:
        folder (str): path to load
        max_files (int): maximum number of files to read, if None this is
            unlimited
    """
    corpora = Corpora()

    files = (glob.glob(os.path.join(path, "*.svg")) +
             glob.glob(os.path.join(path, "*.svg")))[:max_files]
    for i, path in enumerate(files):
        name = os.path.basename(path)
        print("import {:4}/{:4} ({:20})".format(i, len(files), name), end="\r")
        corpus = from_pdf(path)
        corpora[name] = corpus
    return corpora


def sequenced_example(labels, inputs):
    feature_input = [tf.train.Feature(float_list=tf.train.FloatList(value=input_))
                     for input_ in inputs]
    #feature_label = [tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    #                 for label in labels]
    feature_list = {"input": tf.train.FeatureList(feature=feature_input)}
    #                "label": tf.train.FeatureList(feature=feature_label)}
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    features = {"feature":
        {"label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.encode()]))}}
    return tf.train.SequenceExample(feature_lists=feature_lists,
                                    context=features)

def recording(corpus, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for labels, features in corpus:
            example = sequenced_example(labels, features)
            serialized = example.SerializeToString()
            writer.write(serialized)

def create_records(
    output_path,
    ibm_ub_path,
    iam_on_do_path,
    my_handwriting_train_path,
    my_handwriting_test_path,
    load_corpora_from_pickle=False,
    max_files=None):

    pickle_filepath = os.path.join(output_path, "corpora.pkl")
    if not load_corpora_from_pickle:

        corpora_dict = {}
        corpora_dict["ibm"] = corpus_ibm.load(ibm_ub_path,
                                              max_files=max_files)
        corpora_dict["iam_word"], corpora_dict["iam_line"] = \
            corpus_iam.load(iam_on_do_path, max_files=max_files)

        corpora_dict["my_train"] = read_pdf_folder(my_handwriting_train_path,
                                                   max_files=max_files)
        corpora_dict["my_test"] = read_pdf_folder(my_handwriting_test_path,
                                                  max_files=max_files)

        modul_dir, _ = os.path.split(__file__)
        sample_text_path = os.path.join(modul_dir, "data", "sample_text", "The_Zen_of_Python.pdf")
        corpora_dict["zen_test"] = Corpora(
            zen_test=from_pdf(sample_text_path))

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pickle.dump(corpora_dict, open(pickle_filepath, "wb"), -1)
    else:
        corpora_dict = pickle.load(open(pickle_filepath, "rb"))

    iam_test_set = [line.rstrip("\n").replace(".inkml" , "")
                    for line in open(os.path.join(iam_on_do_path, "0.set"), "r")]

    for purpose in ["train", "test"]:
        try:
            os.mkdir(os.path.join(output_path, purpose))
        except FileExistsError:
            pass
    for corpora_name, corpora in corpora_dict.items():
        print("preprocess corpus: {}".format(corpora_name))
        min_words = 2 if ("my" in corpora_name) or ("line" in corpora_name) else 0
        skew_is_horizontal = True if "ibm" in corpora_name else False
        for corpus_name, corpus in corpora.items():
            if corpora_name == "ibm":
                purpose = "test" if corpus_name in ["AARquery_13", "ALSquery_37", "ALSquery_91", "AMCquery_51"] else "train"
            elif corpora_name in ["iam_word", "iam_line"]:
                purpose = "test" if corpus_name in iam_test_set else "train"
            else:
                purpose = "test" if ("test" in corpus_name) else "train"
            directory = os.path.join(output_path, purpose, corpora_name)
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
            record_filename = os.path.join(
                directory, "{}.tfrecord".format(corpus_name))
            if os.path.isfile(record_filename):
                print("continue")
                continue

            labeled_features = preprocessed_corpus(
                corpus, min_words=min_words,
                skew_is_horizontal=skew_is_horizontal)

            recording(labeled_features, record_filename)

def print_record(filename):
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.SequenceExample()
        example.ParseFromString(serialized_example)
        #label = [i.int64_list.value[0] for i in example.feature_lists.feature_list['label'].feature]
        transcription = example.context.feature['label'].bytes_list.value[0].decode()
        if True:  #"M" in transcription:
            features = np.array([i.float_list.value for i in example.feature_lists.feature_list['input'].feature])
            plot_features(features, transcription)
            print("|{}|".format(transcription))
            plt.show()


def main():
    FLAGS = read_flags()
    if FLAGS.show_record != "":
        print_record(FLAGS.show_record)
    else:
        create_records(
            FLAGS.path,
            iam_on_do_path=FLAGS.iam_on_do_path,
            ibm_ub_path=FLAGS.ibm_ub_path,
            my_handwriting_train_path=FLAGS.my_handwriting_train_path,
            my_handwriting_test_path=FLAGS.my_handwriting_test_path,
            load_corpora_from_pickle=FLAGS.load_cache,
            max_files=FLAGS.max_files)

if __name__ == "__main__":
    main()
