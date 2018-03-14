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

import numpy as np

from . import writing


def test_fix_corruption():
    strokes = [np.array([[0, 0], [0, 0], [1, 0]]),
               np.array([[1, 0], [2, 0]]),
               np.array([[2, 1], [3, 1]])]
    ink = writing.Ink.from_corrupted_stroke(strokes)
    assert len(ink.strokes) == 2
    assert len(ink.strokes[0]) == 3
    assert len(ink.strokes[1]) == 2
