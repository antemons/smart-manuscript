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

from . import inkml

def test_order_is_preserved():
    """test whether all strokes are read in preserving order

    Note, that the x-pos of the first point of all strokes increase monotonically
    in not always the case but in the given example it is.
    """
    path = os.path.join(
        os.path.dirname(__file__), 'data', 'sample_text', 'example.inkml')
    ink = inkml.InkML(path).ink()
    for stroke, next_stroke in zip(ink, ink[1:]):
        print(stroke[0][0])
    first_x_pos_of_strokes = [8.35, 10.89, 12.76, 13.06, 14.76, 14.49, 15.3,
                              16.34, 19.15, 19.33, 23.37, 24.75, 25.52, 27.11,
                              30.46, 31.84, 33.78, 33.96, 35.71, 36.76]
    for stroke, correct_x_pos in zip(ink, first_x_pos_of_strokes):
        assert stroke[0][0] == correct_x_pos
