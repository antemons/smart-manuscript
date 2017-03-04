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
from stroke_features import ink_to_features, plot_features
from reader import transcription_to_labels
from handwritten_vector_graphic import load as load_pdf



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
        removed_text = [transcription for transcription, _ in self
                        if (set(transcription) - alphabet_set)]
        print("Removed text:", removed_text)

    def _discard_short_textlines(self):
        min_word_num = FLAGS.min_words_per_line
        self[:] = [(transcription, i) for (transcription, i) in self
                   if len(transcription.split(" ")) > min_word_num]

    def _discard_empty_transcription(self):
        self[:] = [(transcription, i) for (transcription, i) in self
                   if len(transcription) > 0]

    def _discard_single_letters(self):
        self[:] = [(transcription, i) for (transcription, i) in self
                   if len(transcription) > 1]
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
        self._discard_single_letters()
        self._discard_unknown_characters(alphabet)
        self._discard_empty_transcription()
        data = []
        for i, (transcription, ink) in enumerate(self):
            print("{:5.1f}%".format(100 * i / len(self)),
                  flush=True, end="\r")
            features = ink_to_features(ink)
            labels = transcription_to_labels(transcription, alphabet)

            if not features.has_been_well_normalized:
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

    def __getitem__(self, val):
        if val is slice:
            return type(self)(super().__getitem__(val))
        else:
            return super().__getitem__(val)

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
