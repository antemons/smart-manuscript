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

import numpy as np

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


class Transformation(object):

    def __init__(self, param):
        if isinstance(param, np.ndarray):
            param = param.reshape(-1)
        self.matrix = np.stack([param[0:3], param[3:6], [0, 0, 1]])

    @classmethod
    def identity(cls):
        return Transformation([1, 0, 0, 0, 1, 0])

    @classmethod
    def translation(cls, x, y):
        return Transformation([1, 0, x, 0, 1, y])

    @classmethod
    def rotation(cls, angle):
        return Transformation([np.cos(angle), - np.sin(angle), 0,
                               np.sin(angle),   np.cos(angle), 0])

    @classmethod
    def scale(cls, factor):
        if isinstance(factor, tuple):
            return Transformation(
                [factor[0], 0, 0, 0, factor[1], 0])
        else:
            return Transformation([factor, 0, 0, 0, factor, 0])

    @classmethod
    def shear(cls, x_angle=0, y_angle=0):
        return Transformation([1, np.tan(y_angle), 0,
                               np.tan(x_angle), 1, 0])

    @classmethod
    def mirror(cls, angle):
        return Transformation([np.cos(angle),  np.sin(angle), 0,
                               np.sin(angle), -np.cos(angle), 0])

    @property
    def parameter(self):
        return self.matrix.reshape(-1)[[0, 3, 1, 4, 2, 5]]

    @property
    def determinant(self):
        return np.linalg.det(self.matrix)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return (np.dot(other,
                           self.matrix[:2, :2].transpose()) +
                    self.matrix[:2, 2])
        elif isinstance(other, Transformation):
            return Transformation(np.dot(self.matrix,
                                         other.matrix))

    def __invert__(self):
        return Transformation(np.linalg.inv(self.matrix))


def cached_property(function):
    """ decorator to strore the results of a method
    """
    attr_name = '_chached_' + function.__name__

    @property
    def _chached_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)
    return _chached_property
