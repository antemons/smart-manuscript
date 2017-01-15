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

from inkml import plot_ink, read_inkml_file
from tensorflow.python.platform.app import flags
import pylab as plt
import os

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def plot_corpus(corpus, columns=1):
    _, axes_arr = plt.subplots(int(np.ceil(len(corpus))/columns), columns)
    for (transcription, ink), axes in zip(corpus, axes_arr.reshape(-1)):
        plot_ink(ink, transcription, axes)
    plt.show()

FLAGS = flags.FLAGS
flags.DEFINE_string("iam_on_do_path", "data/IAMonDo-db-1.0",
                    "path to IAMonDo-db-1.0 folder (unzipped)")


def import_all_sets(iam_on_db_path):
    corpara_word = {}
    corpara_line = {}
    for i in range(5):
        corpara_word[i], corpara_line[i] = _import_set(iam_on_db_path,
                                                       '%i.set' % i)
    return corpara_word, corpara_line


def _import_set(iam_on_db_path, set_filename):
    print("import set: " + iam_on_db_path + "/" + set_filename)

    def files_in_set(path, set_filename):
        with open(path + "/" + set_filename) as f:
            for line in f:
                yield path + "/" + line.rstrip('\n')

    words = []
    textlines = []
    num_files = 0
    for inkml_path in files_in_set(iam_on_db_path, set_filename):
        if os.path.exists(inkml_path):
            num_files += 1
            new_words, new_textlines = read_inkml_file(inkml_path)
            words.extend(new_words)
            textlines.extend(new_textlines)
    textlines = [(t, i) for (t, i) in textlines
                 if len(t.split(" ")) > FLAGS.min_words_per_line]
    print(" {} words and {} textlines have been imported "
          "from {} inkml-files ".format(len(words), len(textlines), num_files))

    return words, textlines


def main():
    """ load the full IAMonDo database and show random lines and transcription
    """

    import random

    word_sets, line_sets = import_all_sets(FLAGS.iam_on_do_path)
    words, lines = word_sets[1], line_sets[1]
    # words, lines = _import_set(FLAGS.iam_on_do_path, "0.set")

    while True:
        examples = random.sample(words, 8)
        plot_corpus(examples, columns=2)


if __name__ == "__main__":
    main()
