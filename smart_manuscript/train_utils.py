#!/usr/bin/python3

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
from stroke_features import InkFeatures
from reader import transcription_to_labels
from handwritten_vector_graphic import load as load_pdf
import iamondb

FLAGS = flags.FLAGS
flags.DEFINE_integer("min_words_per_line", 3,
                     "num of word required to accept as a textline")
flags.DEFINE_string("iam_on_do_path", "data/IAMonDo-db-1.0",
                    "path to IAMonDo-db-1.0 folder (unzipped)")

class Corpus(list):

    @classmethod
    def from_pdf(cls, filename):
        transcription_filename = os.path.splitext(filename)[0] + ".txt"
        ink = load_pdf(filename)
        with open(transcription_filename) as f:
            transcription = [l.replace("\n", "") for l in f.readlines()]
        assert len(ink.lines) == len(transcription), \
            ink_filename + " %i != %i" % (len(ink.lines),
                                          len(transcription))
        return cls(zip(transcription, ink.lines))

    @classmethod
    def from_folder(cls, folder, filetype=".svg"):
        """ load a corpus by readling all svg in a folder an the corresponding
            transcription in a .txt file with the same basename

        Args:
            folder (str): path to load
        """
        corpus = cls()
        for filename in glob.glob(folder + "*" + filetype):
            new_corpus = cls.from_pdf(filename)
            corpus.extend(new_corpus)
        return corpus

    def _discard_unknown_characters(self, alphabet):
        """ remove in a corpus all entries containing symbols which a not included
            in the alphabet

        Args:
            alphabet (list of str): the alphabet
        """
        alphabet_set = set(alphabet)

        self[:] = [(transcription, i) for (transcription, i) in self
                   if not (set(transcription) - alphabet_set)]

    def _discard_short_textlines(self):
        min_word_num = FLAGS.min_words_per_line
        self[:] = [(transcription, i) for (transcription, i) in self
                   if len(transcription.split(" ")) > min_word_num]

    def _discard_empty_transcription(self):
        self[:] = [(transcription, i) for (transcription, i) in self
                   if len(transcription) > 0]

    @staticmethod
    def _discard_short_features(data):
        data[:] = [(features, labels) for (features, labels) in data
                   if len(features) > 4 and len(features) > len(labels)]

    def convert_to_features_and_labels(self, alphabet,
                                       discard_short_textlines=False):
        """ convert the strokes/transcription of a corpus in a new corpus
            with features/labels
        """


        length_old = len(self)

        if discard_short_textlines:
            self._discard_short_textlines()
        self._discard_unknown_characters(alphabet)
        self._discard_empty_transcription()
        data = []
        for i, (transcription, ink) in enumerate(self):
            print("{:5.1f}%".format(100 * i / len(self)),
                  flush=True, end="\r")
            features = InkFeatures(ink)
            labels = transcription_to_labels(transcription, alphabet)

            if (len(features.features) < 4 or
                    len(labels) > len(features.features) or
                    not features.has_been_well_normalized()):
                continue

            data.append((features.features, labels))

        self._discard_short_features(data)
        length_new = len(data)

        print("Discarded Elements {:5.2f}% ({} / {})".format(
                100 * (length_old - length_new) / length_old,
                length_old - length_new, length_old))

        return data

    def plot_sample(self, rows=4, columns=2):
        _, axes_arr = plt.subplots(rows, columns)
        for axes in axes_arr.reshape(-1):
            transcription, ink = choice(self)
            ink.plot_pylab(axes, transcription)
        plt.show()


class Corpora(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_features_and_labels(self, alphabet):

        data = {}
        for i, (key, corpus) in enumerate(self.items()):
            print("convert {} ({}/{})".format(key, i, len(self)))
            data[key] = corpus.convert_to_features_and_labels(
                alphabet, discard_short_textlines=("line" in key))
        return data


def load_corpora():
    """ load all coporas """
    corpora = Corpora()

    corpara_word, corpara_line = iamondb.load(FLAGS.iam_on_do_path)
    corpora["train_word"] = Corpus(sum(corpara_word[:4], []))
    corpora["test_word"] = corpara_word[4]
    corpora["train_line"] = Corpus(sum(corpara_line[:4], []))
    corpora["test_line"] = corpara_line[4]

    if FLAGS.train_my_writing:
        folder="data/my_handwriting/"
        corpus_train = Corpus.from_folder(folder + "train/")
        corpus_test = Corpus.from_folder(folder + "test/")
        corpora["train_my_writing"] = corpus_train
        corpora["test_my_writing"] = corpus_test

    if FLAGS.test_the_zen_of_python:
        corpora["test_the_zen_of_python"] = \
            Corpus.from_pdf("sample_text/The_Zen_of_Python.svg")

    return corpora


def get_test_batches(data):
    batches = {}

    batches["word"] = sample(data["test_word"], FLAGS.test_batch_size)

    if FLAGS.train_textlines:
        batches["line"] = sample(data["test_line"],
                                 FLAGS.test_batch_size_lines)
    if FLAGS.train_my_writing:
        batches["my_writing"] = sample(data["test_my_writing"],
                                       FLAGS.test_batch_size_lines)

    return batches


def get_trainings_batch_creator(data):
    if FLAGS.train_textlines and FLAGS.train_my_writing:
        def batch_train(_):
            batch_size_lines = 5
            batch_my_writing = 1
            batch_size_words = (FLAGS.batch_size - batch_size_lines -
                                batch_my_writing)
            if np.random.random() < 0.25:
                return (
                    sample(data["train_word"], batch_size_words) +
                    sample(data["train_line"], batch_size_lines) +
                    sample(data["train_my_writing"], batch_my_writing))
            else:
                return (
                    sample(data["train_word"], batch_size_words) +
                    sample(data["train_line"], batch_size_lines + 1))
    elif not FLAGS.train_textlines and not FLAGS.train_my_writing:
        def batch_train(_):
            return data["test_word"].next_batch(FLAGS.batch_size)
    elif FLAGS.train_textlines and not FLAGS.train_my_writing:
        def batch_train(_):
            batch_size_lines = 6
            batch_size_words = (FLAGS.batch_size - batch_size_lines)
            return (
                sample(data["train_word"], batch_size_words) +
                sample(data["train_line"], batch_size_lines))
    else:
        raise Exception("not implemented")
    return batch_train
