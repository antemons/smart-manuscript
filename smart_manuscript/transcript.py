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


from tensorflow.python.platform.app import flags
from handwritten_vector_graphic import load as load_svg
from reader import Reader

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "file", "sample_text/The_Zen_of_Python.svg", "svg-file to transcript")


def main():
    """ transcript the a svg-file and print the transcription

        The filepath is FLAGS.file
    """
    reader = Reader(path=FLAGS.graph_path)
    ink = load_svg(FLAGS.file)
    transcription = reader.recognize_page(ink)
    print("Transcription:\n")
    print(transcription)

if __name__ == "__main__":
    main()
