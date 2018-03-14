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
from .reader import Reader
from .inkml import InkML
from .writing import Ink
from .utils import Transformation, colored_str_comparison
import pylab as plt

def test_recognition():
    CORRECT_TEXT = "Machine learning rocks !"
    model_path = os.path.join(
        os.path.dirname(__file__), 'data', 'model', 'model.ckpt')
    recognizer = Reader(model_path)
    inkml_path = os.path.join(
        os.path.dirname(__file__), 'data', 'sample_text', 'example.inkml')
    inkml = InkML(inkml_path, flip_y=True)
    ink = Ink(inkml.ink())
    recognized_text, *_ = recognizer.recognize_line(ink.strokes)
    print(colored_str_comparison(recognized_text, CORRECT_TEXT))
    assert recognized_text == CORRECT_TEXT
