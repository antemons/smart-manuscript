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
from transcript import main as transcript
from application import main as application

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "transcript", False, "transcript (or use transcript.py)")


def main():
    """ choose either the transcription of a handwritten file or the
        handwritten input
    """
    if FLAGS.transcript:
        transcript()
    else:
        application()


if __name__ == "__main__":
    main()
