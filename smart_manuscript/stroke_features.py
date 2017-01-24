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
import pylab as plt
from itertools import groupby, accumulate
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.optimize import minimize
from copy import deepcopy
from utils import Transformation, cached_property

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


class Ink(object):
    """ Set of strokes

        nested list strokes (float[L,2]), the (x,y)-points for each point in
        each stroke in each strokes
    """

    def __init__(self, strokes):
        strokes = deepcopy(strokes)
        # TODO(dv): remove already here the dublicated points
        # strokes = self._remove_dublicate_points(strokes)
        self._connect_gapless_strokes(strokes)
        self._concatenated_strokes = np.concatenate(strokes)
        sections = np.cumsum([len(s) for s in strokes])[:-1]
        self.strokes = np.split(self._concatenated_strokes, sections)

    def __deepcopy__(self, _):
        return self.__class__(self.strokes)  # connect_gaps and remove duplicated redundant

    def __iter__(self):
        return self.strokes.__iter__()

    @property
    def concatenated_strokes(self):
        return self._concatenated_strokes

    @concatenated_strokes.setter
    def concatenated_strokes(self, value):
        self._concatenated_strokes[:] = value

    @staticmethod
    def _remove_dublicate_points(strokes):
        def remove_dublicates_from_stroke(stroke):
            return np.array([x for x, _ in groupby(stroke, tuple)])
        return [remove_dublicates_from_stroke(stroke)
                for stroke in strokes]

    @staticmethod
    def _connect_gapless_strokes(strokes):
        """ connect strokes in the the consecutive stroke starts where the
            proceding stroke ended
        """
        for i in list(range(len(strokes)-1))[::-1]:
            if all(strokes[i][-1] == strokes[i+1][0]):
                strokes[i] = np.concatenate([strokes[i][:-1], strokes[i+1]])
                strokes.pop(i+1)

    @property
    def boundary_box(self):
        return (min(self._concatenated_strokes[:, 0]),
                max(self._concatenated_strokes[:, 0]),
                min(self._concatenated_strokes[:, 1]),
                max(self._concatenated_strokes[:, 1]))

    @property
    def length(self):
        def stroke_length(stroke):
            x = stroke[:, 0]
            y = stroke[:, 1]
            dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
            return sum(dist)

        return sum(stroke_length(stroke) for stroke in self)

    def transform(self, transformation):
        self.concatenated_strokes = transformation * self.concatenated_strokes


class InkPage(Ink):
    """ Ink on a page """

    def __init__(self, strokes, page_size=None):
        super().__init__(strokes)
        self.page_size = page_size

    def __deepcopy__(self, _):
        return self.__class__(self.strokes, page_size=self.page_size)

    @cached_property
    def lines(self, min_seperation=None):  # TODO(daniel): remove min_seperation
        """ split the ink in several lines
        """
        lines = []
        min_x, max_x, _, _ = self.boundary_box
        if min_seperation is None:
            min_seperation = (max_x - min_x)/10
        for stroke in self:
            if (not len(lines) or (lines[-1][-1][-1, 0] -
                                   min_seperation > stroke[0, 0])):
                lines.append([])
            lines[-1].append(stroke)
        return [Ink(line) for line in lines]


class InkNormalization(object):
    """ normalize a stroke by slant, skew, baseline, height, and width
    """

    def __init__(self, ink):
        self.ink = deepcopy(ink)
        self._transformation = Transformation.identity()
        self.__strokes_orig = deepcopy(self.ink)
        self.normalize_skew()
        self.__strokes_skew = deepcopy(self.ink)
        self.normalize_slant()
        self.__strokes_slant = deepcopy(self.ink)
        self.normalize_baseline()
        self.__strokes_baseline = deepcopy(self.ink)
        self.normalize_width()
        self.normalize_left()
        self.__strokes_width = deepcopy(self.ink)

    def plot(self):
        _, axes = plt.subplots(5)
        self._plot_strokes(self.__strokes_orig, "original", axes[0])
        width = self.ink.boundary_box[1]
        box = np.array([[0,0], [width, 0], [width, 1], [0, 1]])
        axes[0].scatter(*((~self._transformation) * box).transpose())

        self._plot_strokes(self.__strokes_skew, "normalize skew", axes[1])
        axes[1].plot(self.__strokes_skew.boundary_box[:2], [0, 0], 'k:')

        self._plot_strokes(self.__strokes_slant, "normalize slant", axes[2])

        self._plot_strokes(
            self.__strokes_baseline, "normalize baseline", axes[3])
        axes[3].scatter(*self.__minima.transpose(), c='b', edgecolors='face')
        axes[3].scatter(*self.__maxima.transpose(), c='r', edgecolors='face')
        axes[3].plot(self.__strokes_baseline.boundary_box[:2], [0, 0], 'k:')
        axes[3].plot(self.__strokes_baseline.boundary_box[:2], [1, 1], 'k:')

        self._plot_strokes(self.__strokes_width, "normalize width", axes[4])
        axes[4].plot(self.__strokes_width.boundary_box[:2], [0, 0], 'k:')
        axes[4].plot(self.__strokes_width.boundary_box[:2], [1, 1], 'k:')
        plt.show()

    def _transform(self, transformation):
        self._transformation = transformation * self._transformation
        self.ink.transform(transformation)

    @staticmethod
    def _plot_strokes(strokes, description, axes=None):
        """ plot current state of normalization

        Args:
            strokes (Ink): ink which is shown
            description (str): title
        """
        if axes is not None:
            axes.set_title(description)
            for stroke in strokes:
                axes.plot(stroke[:, 0], stroke[:, 1], 'g-')
            axes.set_aspect('equal')

    def normalize_skew(self):
        """ normalize skew by an linear fit

        Args:
            strokes (Ink): ink to normalize
        """
        # TODO: only if long enough

        a, b = np.polyfit(self.ink.concatenated_strokes[:, 0],
                          self.ink.concatenated_strokes[:, 1], 1)
        transformation = (Transformation.rotation(- np.arctan(a)) *
                           Transformation.translation(0, -b))
        #transformation = (Transformation.shear(y_angle=-np.arctan(a)) *
        #                  Transformation.translation(0, -b))
        self._transform(transformation)

    def normalize_left(self):
        """ set the leftmost point to x = 0
        """
        min_x = self.ink.boundary_box[0]
        transformation = Transformation.translation(- min_x, 0)
        self._transform(transformation)

    def normalize_slant(self):
        """ normalize the slant
        """

        def tilted_ink_length(angle):
            ink = deepcopy(self.ink)
            ink.transform(Transformation.shear(y_angle=angle))
            return ink.length

        angle = minimize(tilted_ink_length, 0).x
        transformation = Transformation.shear(y_angle=angle)
        self._transform(transformation)


    def normalize_baseline(self):
        """ normalize baseline to y = 0 and mean line / height to y = 1

        Fits the local minima (maxima) to the baseline (mean line), resp.

        Args:
            strokes (Ink): ink to normalize
        """

        minima = []
        maxima = []
        for stroke in self.ink:
            # TODO(dv): avoid several consecutives extrema
            #           (e.g. when maximum is divide)
            idx_min = argrelextrema(stroke[:, 1], np.less_equal)[0]
            idx_max = argrelextrema(stroke[:, 1], np.greater_equal)[0]
            if len(stroke) == 1:
                idx_max = np.concatenate([[0], idx_max])
                idx_min = np.concatenate([[0], idx_min])
            else:
                if stroke[0, 1] < stroke[1, 1]:
                    idx_min = np.concatenate([[0], idx_min])
                else:
                    idx_max = np.concatenate([[0], idx_max])
                if stroke[-1, 1] < stroke[-2, 1]:
                    idx_min = np.concatenate([idx_min, [-1]])
                else:
                    idx_max = np.concatenate([idx_max, [-1]])
            minima.extend(stroke[idx_min])
            maxima.extend(stroke[idx_max])
        minima, maxima = np.array(minima), np.array(maxima)
        assert (minima.size and maxima.size)

        BASELINE = np.average(minima[:, 1])
        MEANLINE = np.average(maxima[:, 1])
        if MEANLINE > BASELINE:
            minima = minima[minima[:, 1] < MEANLINE]
            maxima = maxima[maxima[:, 1] > BASELINE]
            BASELINE = np.average(minima[:, 1])
            MEANLINE = np.average(maxima[:, 1])
            HEIGHT = MEANLINE - BASELINE
            transformation = (Transformation.scale(1 / HEIGHT) *
                              Transformation.translation(0, - BASELINE))
        else:
            transformation = Transformation.translation(0, - BASELINE + .5)

        self._transform(transformation)

        self.__minima = transformation * minima
        self.__maxima = transformation * maxima

    def normalize_width(self):
        """ normalize the width by using the average width between two
            intersections with line between base and mean line

        Args:
            strokes (Ink): ink to normalize
        """
        intersections = 0
        for stroke in self.ink:
            is_below = stroke[:, 1] < 0.5
            is_below_merged = [x for x, _ in groupby(is_below)]
            intersections += len(is_below_merged) - 1

        # TODO(dv): replace definition before training next graph
        # is_below = strokes.concatenated_strokes[:, 1] < 0.5
        # intersections = len([x for x, _ in groupby(is_below)])

        min_x, max_x, _, _ = self.ink.boundary_box
        width = max_x - min_x
        scale_width = intersections / (2 * width) if width else 1
        transformation = Transformation.scale((scale_width, 1))
        self._transform(transformation)


class InkFeatures(object):
    """ Generates features of a given stroke
    """

    NUM_FEATURES = 14
    DOTS_PER_UNIT = 5
    RDP_EPSILON = 0.02

    def __init__(self, strokes, normalize=True):
        """ Provide a list of features characterizing the set of strokes

        Args:
            strokes (Ink): ink to normalize
        """
        if normalize:
            strokes = InkNormalization(strokes).ink
        # strokes = deepcopy(strokes)
        #self.ink = strokes
        self.ink = self._remove_dublicate_points(strokes)

    # TODO(daniel): move to Ink
    @staticmethod
    def _remove_dublicate_points(strokes):
        def remove_dublicate_from_stroke(stroke):
            return np.array([x for x, _ in groupby(stroke, tuple)])
        strokes = [remove_dublicate_from_stroke(stroke) for stroke in strokes]
        return strokes


    @cached_property
    def _splines(self):
        return [self._create_spline(stroke) for stroke in self.ink]

    @cached_property
    def _knots(self):
        return [self._get_knots(spline) for spline in self._splines]

    def _plot_encased(self, axes):
        axes.set_aspect('equal')
        axes.set_title('encased')
        color_dict = {(1, 1): 'r', (0, 1): 'y', (1, 0): 'k', (0, 0): 'g'}
        colors = [color_dict[tuple(e)]
                  for e in self.encased]
        axes.scatter(self.x_position, self.y_position,
                     c=colors,  edgecolors='face')

    def _plot_pressure(self, axes):
        axes.set_aspect('equal')
        axes.set_title('pressure')
        color_dict = {(1, 1): 'w', (0, 1): 'y', (1, 0): 'k'}
        colors = [color_dict[tuple(p)] for p in self.pressure]
        axes.scatter(self.x_position, self.y_position,
                     c=colors,  edgecolors='face')

    def _plot_curvature(self, axes):
        axes.set_aspect('equal')
        axes.set_title('curvature')

        def circle(x, y, radius):
            theta = np.linspace(0, 2 * np.pi, 100)
            axes.plot(x+radius * np.sin(theta), y+radius * np.cos(theta), 'y:')

        for i in range(len(self.features)):
            radius = self.radius_of_curvature[i]
            circle(self._connected_stroke[i][0] - self._tangent[i][1] * radius,
                   self.y_position[i] + self._tangent[i][0] * radius,
                   radius)

    def _plot_rough_tangent(self, axes):
        axes.set_aspect('equal')
        axes.set_title('rough_tangent')

        def line(x, y, dx, dy):
            axes.plot([x, x+dx], [y, y+dy], '->')

        for i in range(len(self.features)):
            line(self._connected_stroke[i][0], self.y_position[i],
                 self._rough_tangent[i][0], self._rough_tangent[i][1])

    def _plot_pos_between(self, axes):
        axes.set_aspect('equal')
        axes.set_title('pos_between')
        axes.plot(self.pos_between[:, 0], self.pos_between[:, 1], "ko")

    def _plot_x_low_pass(self, axes):
        axes.set_aspect('equal')
        axes.set_title('x_low_pass')
        axes.plot(
            self.x_low_pass, np.linspace(2, 3, len(self.x_position)), ":")
        axes.plot(
            self.x_low_pass + self.x_low_pass_filtered,
            np.linspace(2, 3, len(self.x_position)), ":")
        axes.plot(
            self.x_low_pass_filtered, np.linspace(
                2, 3, len(self.x_position)), ":")

    def _plot_intersection(self, axes):
        axes.set_aspect('equal')
        axes.set_title('intersection')
        axes.scatter(self.x_position[self.intersection == 1],
                     self.y_position[self.intersection == 1])

    def _plot_data(self, axes, data, title="data"):
        axes.set_aspect('equal')
        axes.set_title(title)
        axes.scatter(
            self.x_position, self.y_position,
            c=data, cmap=plt.cm.get_cmap('bwr'), edgecolors='face')

    def _plot_splines(self, axes):
        axes.set_aspect('equal')
        for spline in self._splines:
            u = np.linspace(0, spline[0][-1], 1000*spline[0][-1])
            stroke = np.array(interpolate.splev(u, spline)).transpose()
            axes.plot(stroke[:, 0], stroke[:, 1], "g-")

    def _plot_orig_strokes(self, axes):
        axes.set_aspect('equal')
        axes.set_title("orig_strokes")
        for stroke in self.ink:
            axes.plot(stroke[:, 0], stroke[:, 1], "bo")

    def plot_all(self):
        """ plot the stroke
        """
        _, subplots = plt.subplots(4, 2)
        for axes in subplots.reshape(-1):
            self._plot_splines(axes)
        self._plot_orig_strokes(subplots[0, 0])
        self._plot_encased(subplots[1, 0])
        self._plot_intersection(subplots[2, 0])
        self._plot_curvature(subplots[3, 0])
        self._plot_pressure(subplots[0, 1])
        plt.show()

    def _create_spline(self, stroke):
        """ create the spline from the stroke
        """

        assert len(stroke) > 0
        if len(stroke) < 4:
            stroke_new = np.zeros([4, 2])
            stroke_new[:] = np.outer(np.linspace(0, 1, 4), stroke[0])\
                + np.outer(np.linspace(1, 0, 4), stroke[-1])
            if (stroke[0] == stroke[-1]).all:
                stroke_new[:, 1] += - np.linspace(0, -0.1, 4)
            stroke = stroke_new
        x = stroke[:, 0]
        y = stroke[:, 1]
        dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        try:
            spline, _ = interpolate.splprep([x, y], k=3, u=dist_along, s=0)
        except SystemError:
            print(x, y, type(x))
            raise
        return spline

    @cached_property
    def _rough_tangent(self):
        """ tangent given on a coarse grained polygon
        """
        rough_tangent = []
        for knots, spline in zip(self._knots, self._splines):
            knots_between = np.zeros(len(knots)+1)
            knots_between[0] = knots[0]
            knots_between[-1] = knots[-1]
            knots_between[1:-1] = (knots[:-1] + knots[1:])/2
            pos_between = np.array(
                interpolate.splev(knots_between, spline)).transpose()
            self.pos_between = pos_between
            dpos_between = pos_between[1:]-pos_between[:-1]
            dpos_between = dpos_between /\
                (np.sum(dpos_between**2, axis=1)**0.5)[:, np.newaxis]
            rough_tangent += dpos_between.tolist()
        return np.array(rough_tangent)

    @cached_property
    def _length_along(self):
        """ spline length up to the knot
        """
        return np.concatenate(self._knots)

    def _get_knots(self, spline):  # _get_knots_equidistant
        """ discretize the spline and try have DOTS_PER_UNIT dot per unit length
        """
        len_of_spline = spline[0][-1]
        knots_of_spline = max(2, self.DOTS_PER_UNIT*len_of_spline)
        u = np.linspace(0, len_of_spline, knots_of_spline)
        return u

    def _get_knots_rdp(self, spline):
        # TODO: use shapely.geometry.simplify instead of rdp
        # this function is unsed (replaced by _get_knots_equidistant)
        from rdp import rdp
        u = np.linspace(0, spline[0][-1], max(2, 100*spline[0][-1]))
        stroke = np.array(interpolate.splev(u, spline)).transpose()
        intrpl_stroke = rdp(stroke, epsilon=self.RDP_EPSILON)
        intrpl_stroke_args = [
            (stroke[:, 0]+1j*stroke[:, 1]).tolist().index(dot[0]+1j*dot[1])
            for dot in intrpl_stroke]
        return u[intrpl_stroke_args]

    @property
    def y_position(self):
        """ y-coordinate
        """
        return self._connected_stroke[:, 1]

    @property
    def x_position(self):
        """ x-coordinate
        """
        return self._connected_stroke[:, 0]

    @cached_property
    def _connected_stroke(self):
        """ x,y position of the splines (discretized)
        """
        stroke = []
        for spline, knots in zip(self._splines, self._knots):
            stroke += list(zip(*interpolate.splev(knots, spline)))
        return np.array(stroke)

    @cached_property
    def _stroke_der(self):
        """ derivative of the spline
        """
        stroke_der = []
        for spline, knots in zip(self._splines, self._knots):
            stroke_der += list(zip(*interpolate.splev(knots, spline, der=1)))
        return np.array(stroke_der)

    @cached_property
    def _stroke_cur(self):
        """ curvature of the spline
        """
        stroke_cur = []
        for i in range(len(self._splines)):
            stroke_cur += np.array(
                interpolate.splev(self._knots[i], self._splines[i],
                                  der=2)).transpose().tolist()
        return np.array(stroke_cur)

    @cached_property
    def _tangent(self):
        """ normalized slope
        """
        return (self._stroke_der /
                np.sum(self._stroke_der**2, axis=1)[:, np.newaxis]**0.5)

    @cached_property
    def radius_of_curvature(self):
        """ radius of a tangent circle at the point (positive and negaive)
        """
        with np.errstate(divide='ignore'):
            ret = (np.sum(self._stroke_der**2, axis=1)**(3/2) /
                   (self._stroke_der[:, 0] * self._stroke_cur[:, 1] -
                    self._stroke_der[:, 1] * self._stroke_cur[:, 0]))
        return ret

    @cached_property
    def radius_normalized(self):
        """ a function of the radius
        """
        return (np.sign(self.radius_of_curvature) /
                (1 + np.abs(self.radius_of_curvature)))

    @cached_property
    def delta_y(self):
        """ difference to the y-coordinate to the preceding point
        """
        return self._connected_stroke[1:, 1] - self._connected_stroke[:-1, 1]

    @cached_property
    def delta_y_extended(self):
        """ difference to the y-coordinate to the preceding point and zero for
            the first point
        """
        return np.concatenate([[0], self.delta_y])

    @cached_property
    def delta_x(self):
        """ difference to the x-coordinate to the preceding point
        """
        return self._connected_stroke[1:, 0]-self._connected_stroke[:-1, 0]

    @cached_property
    def delta_x_extended(self):
        """ difference to the x-coordinate to the preceding point and zero for
            the first point
        """
        return np.concatenate([[0], self.delta_x])

    @cached_property
    def pressure(self):
        """ Whether at the point the stroke starts, ends, or continues
        """
        cuts = [len(knots) for knots in self._knots]
        pressure = np.ones([sum(cuts), 2])
        for i in accumulate(cuts):
            pressure[i-1, 1] = 0
            pressure[i % len(pressure), 0] = 0
        return pressure

    @cached_property
    def _angle_direction(self):
        """ The angle of the current writing and the horizontal line
        """
        alpha = np.arctan2(self._tangent[:, 1], self._tangent[:, 0])
        return alpha

    @cached_property
    def writing_direction(self):
        """ The vector representation of _angle_direction
            (to avoid jump from 2*pi to 0)
        """
        return np.column_stack([np.cos(self._angle_direction),
                                np.sin(self._angle_direction)])

    @cached_property
    def _angle_curvature(self):
        """ The difference of to consecutives angles giving the writing direction
        """
        tmp = np.zeros_like(self._angle_direction)
        tmp[1:-1] = self._angle_direction[2:]-self._angle_direction[1:-1]
        return tmp

    @cached_property
    def x_low_pass(self):
        """ The average movement with respect to each single stroke
        """
        low_pass = np.zeros_like(self.x_position)
        begin = 0
        for i, knots in enumerate(self._knots):
            end = begin + len(knots)
            a, b = np.polyfit(
                self._length_along[begin:end], self.x_position[begin:end], 1)
            low_pass[begin:end] = a * self._length_along[begin:end] + b
            begin = end
        assert end == len(self.x_position)
        return low_pass

    @cached_property
    def x_low_pass_filtered(self):
        """ The deviation from the mean average movement in each stroke
        """
        return self.x_position - self.x_low_pass

    @cached_property
    def curvature(self):
        """ The vector representation of _angle_curvature
            (to avoid jump from 2*pi to 0)
        """
        return np.sin(self._angle_curvature), np.cos(self._angle_curvature)

    @cached_property
    def intersection(self):
        """ Whether an intersection occurs in the neightbourhood
        """

        phi = np.arctan2(self.delta_x, self.delta_y)
        x_diffs = (self.x_position[np.newaxis, :] -
                   self.x_position[:-1, np.newaxis])
        y_diffs = (self.y_position[np.newaxis, :] -
                   self.y_position[:-1, np.newaxis])
        phi_diffs = (phi[:, np.newaxis] - np.arctan2(x_diffs, y_diffs))

        relative_orientation = phi_diffs % (2 * np.pi) < np.pi

        # two consecutives points are on different side of the reference line
        # given by two consecutives points
        conditions = relative_orientation[:, :-1] ^ relative_orientation[:, 1:]

        # exclude intersections with the "gaps"
        conditions = (
            conditions & (self.pressure[:-1, 1]*self.pressure[1:, 0] == 1))

        # exclude self intersections and intersections with neightbours
        np.fill_diagonal(conditions, False)
        np.fill_diagonal(conditions[:-1, 1:], False)
        np.fill_diagonal(conditions[1:, :-1], False)

        # and vise versa
        conditions = conditions & conditions.transpose()
        conditions = conditions.any(0)
        ret = np.zeros_like(self.x_position, bool)
        ret[1:] += ret[1:] | conditions
        ret[:-1] += ret[:-1] | conditions
        return 1*ret

    @cached_property
    def encased(self):
        """ Whether the point is encased (from top or bottom)
        """
        x = self.x_position
        y = self.y_position
        with np.errstate(divide='ignore', invalid='ignore'):
            # TODO(dv): avoid invalid error here
            a = self.delta_y / self.delta_x
        with np.errstate(invalid='ignore'):
            b = y[:-1] - a * x[:-1]
            point_is_above = (
                y[:, np.newaxis] > a[np.newaxis, :]*x[:, np.newaxis] +
                b[np.newaxis, :])
        conditions = (self.pressure[:-1, 1] *
                      self.pressure[1:, 0] == 1)[np.newaxis, :]

        tmp = ((x[:, np.newaxis] - x[np.newaxis, :-1]) *
               (x[:, np.newaxis] - x[np.newaxis, 1:])) < 0
        encased_from_below = 1*np.any(point_is_above*tmp*conditions, axis=1)
        encased_from_top = 1*np.any((- point_is_above)*tmp*conditions, axis=1)
        encased = np.column_stack([encased_from_below, encased_from_top])
        return encased

    @cached_property
    def encased_old(self):
        """ see encased, but is too slow due to sproot
        """
        def shift_spline(spline, shift_vec):
            spline_coefficients = np.array(spline[1])
            return [
                spline[0], [
                    spline_coefficients[0] + shift_vec[0],
                    spline_coefficients[1] + shift_vec[1]],
                spline[2]]

        encased_from_top = []
        encased_from_below = []
        idx = 0
        TOLERANCE_THERESHOLD = 10**-2

        for knots in self._knots:
            for _ in knots:
                ys = []
                for spline_compare in self._splines:
                    spline_shifted = shift_spline(
                        spline_compare, [-self.x_position[idx], 0])
                    roots = interpolate.sproot(spline_shifted, mest=10)
                    if len(roots[0]) > 0:
                        ys.extend(interpolate.splev(
                            roots[0], spline_compare)[1])

                encased_from_top.append(
                    1*any(y > self.y_position[idx] + TOLERANCE_THERESHOLD
                          for y in ys))
                encased_from_below.append(
                    1*any(y < self.y_position[idx] - TOLERANCE_THERESHOLD
                          for y in ys))
                idx += 1
        return encased_from_below, encased_from_top

    @cached_property
    def features(self):
        """ provide a collection of features

        Returns:
            array[N, NUM_FEATURES] of float: the features
        """
        features = np.column_stack([
            self.x_low_pass_filtered,
            self.y_position,
            self.pressure[:, 0],
            self.pressure[:, 1],
            self.radius_normalized,
            self.writing_direction[:, 0],
            self.writing_direction[:, 1],
            self._rough_tangent[:, 0],
            self._rough_tangent[:, 1],
            self.encased[:, 0],
            self.encased[:, 1],
            self.delta_x_extended,
            self.delta_y_extended,
            # TODO(dv): self.closest_point ?
            self.intersection])
        assert self.NUM_FEATURES == features.shape[1]
        return features


def main():
    from handwritten_vector_graphic import load
    from tensorflow.python.platform.app import flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "file", "sample_text/The_Zen_of_Python.pdf",
        "file to show features (either PDF or SVG)")
    flags.DEFINE_integer(
        "line_num", 0,
        "select line number")

    ink_page = load(FLAGS.file)
    ink = ink_page.lines[FLAGS.line_num]
    normalized = InkNormalization(ink)
    normalized.plot()
    strokes_features = InkFeatures(normalized.ink, normalize=False)
    strokes_features.plot_all()

if __name__ == "__main__":
    main()
