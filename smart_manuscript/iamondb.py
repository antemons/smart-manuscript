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

import inkml
from tensorflow.python.platform.app import flags
import numpy as np
import pylab as plt
import os
import train_utils

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"



def load(iam_on_db_path):
    corpara_word = []
    corpara_line = []
    for i in range(5):
        word_corpus, line_corpus = _import_set(iam_on_db_path, '%i.set' % i)
        corpara_word.append(word_corpus)
        corpara_line.append(line_corpus)
    return corpara_word, corpara_line


def _import_set(iam_on_db_path, set_filename):
    print("import set: " + iam_on_db_path + "/" + set_filename)

    def files_in_set(path, set_filename):
        with open(path + "/" + set_filename) as f:
            for line in f:
                yield path + "/" + line.rstrip('\n')

    corpara_word = train_utils.Corpus()
    corpara_line = train_utils.Corpus()
    num_files = 0
    filenames = list(files_in_set(iam_on_db_path, set_filename))[:25]
    for i, inkml_filename in enumerate(filenames):
        print("import {:3}/{:3} ({})  ".format(
            i, len(filenames), inkml_filename), end="\r")
        if os.path.exists(inkml_filename):
            num_files += 1
            inkml_file = inkml.InkML(inkml_filename)
            corpara_word.extend(inkml_file.words)
            corpara_line.extend(inkml_file.lines)
    print(" {} words and {} textlines have been imported "
          "from {} inkml-files ".format(len(corpara_word), len(corpara_line),
                                        num_files))

    return corpara_word, corpara_line


def main():
    """ load the full IAMonDo database and show random lines and transcription
    """



    import random
    # word_sets, line_sets = import_all_sets(FLAGS.iam_on_do_path)
    # words, lines = word_sets[1], line_sets[1]
    FLAGS = flags.FLAGS
    # flags.DEFINE_string("iam_on_do_path", "data/IAMonDo-db-1.0",
    #                     "path to IAMonDo-db-1.0 folder (unzipped)")

    words, lines = _import_set(FLAGS.iam_on_do_path, "0.set")

    while True:
        lines.plot_sample()


if __name__ == "__main__":
    main()
