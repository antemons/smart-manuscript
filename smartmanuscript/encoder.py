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

import pickle
import os.path

ALPHABET = list("abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "1234567890 ,.:;*+()/!?+-'\"$")


class Encoder:
    class SymbolNotInAlphabet(KeyError):
        pass
    """ convert string to list of int and vise versa via its alphabet
    """
    FILE = "alphabet.pkl"

    def __init__(self, alphabet):
        self._alphabet = alphabet

    @classmethod
    def from_file(cls, path):
        file_path = os.path.join(path, cls.FILE)
        alphabet = pickle.load(open(file_path, "rb"))
        return cls(alphabet)

    def save(self, path):
        file_path = os.path.join(path, self.FILE)
        pickle.dump(self._alphabet, open(file_path, "wb"), -1)

    @property
    def alphabet(self):
        return self._alphabet

    def decode(self, labels):
        """ convert labels (list of int) to transcription (str)
        """
        return "".join([self._alphabet[n] for n in labels])

    def encode(self, transcription):
        """ convert transcription (str) to labels (list of int)
        """
        try:
            return [self._alphabet.index(c) for c in transcription]
        except ValueError:
            raise self.SymbolNotInAlphabet

encoder = Encoder(ALPHABET)
