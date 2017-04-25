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

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from .handwritten_vector_graphic import load as ink_from_file
from .reader import Reader
from .searchable_pdf import SearchablePDF

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def read_flags():
    parser = argparse.ArgumentParser(
        description='Transcribe (digitize) handwritten manuscripts and digitizer pen input')
    parser.add_argument('source', help='the manuscript (pdf or svg) to transcribe')
    parser.add_argument(
        'target', nargs='?',
        default="transcribed_manuscript.pdf",
        help='the generated transcribed manuscipt')
    dafault_model_path = os.path.join(
        os.path.dirname(__file__), 'data', 'model', 'model.ckpt')
    parser.add_argument('--model_path', default=dafault_model_path,
                        help='the tensorflow model')
    parser.add_argument('--dry-run', action='store_true',
                        help='no output is written')

    return parser.parse_args()


def main():
    """ Transcribe (digitize) handwritten manuscripts and digitizer pen input
    """

    args = read_flags()
    source_manuscript = args.source
    target_manuscript = args.target


    searchable_pdf = SearchablePDF(args.model_path)
    if args.dry_run:
        reader = Reader(model_path=FLAGS.model_path)
        strokes, _ = ink_from_file(FLAGS.file)
        transcription = reader.recognize_page(strokes)
    else:
        searchable_pdf.generate(source_manuscript, target_manuscript)


if __name__ == "__main__":
    main()
