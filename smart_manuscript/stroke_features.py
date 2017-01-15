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
from scipy import optimize, interpolate
from scipy.signal import argrelextrema

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"


def lazyprop(function):
    """ decorator to strore the results of a method
    """
    attr_name = '_lazy_' + function.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)
    return _lazyprop


def split_lines(ink_of_page, min_seperation=None):
    """ split the ink in several lines
    """
    ink_of_lines = []
    min_x, max_x, _, _ = boundary_box(ink_of_page)
    if min_seperation is None:
        min_seperation = (max_x - min_x)/10
    for stroke in ink_of_page:
        if (not len(ink_of_lines) or
                ink_of_lines[-1][-1][-1, 0] - min_seperation > stroke[0, 0]):
            ink_of_lines.append([])
        ink_of_lines[-1].append(stroke)
    return ink_of_lines


def boundary_box(strokes):
    try:
        connected_strokes = np.concatenate(strokes)
    except:
        print(strokes)
        raise
    return (min(connected_strokes[:, 0]), max(connected_strokes[:, 0]),
            min(connected_strokes[:, 1]), max(connected_strokes[:, 1]))


class NormalizeStroke(object):
    """ normalize a stroke by slant, skew, baseline, height, and width
    """

    def __init__(self, strokes):
        self.__strokes_skew = self.normalize_skew(strokes)
        self.__strokes_slant = self.normalize_slant(self.__strokes_skew)
        self.__strokes_baseline, self.__minima, self.__maxima = \
            self.normalize_baseline(self.__strokes_slant)
        self.__strokes_width = self.normalize_width(self.__strokes_baseline)
        self._strokes = self.__strokes_width

    def plot_normalization(self):
        _, axes = plt.subplots(4)
        self._plot_strokes(self.__strokes_skew, "normalize skew", axes[0])
        self._plot_strokes(self.__strokes_slant, "normalize slant", axes[1])
        self._plot_strokes(
            self.__strokes_baseline, "normalize baseline", axes[2])
        axes[2].scatter(*self.__minima.transpose(), c='b', edgecolors='face')
        axes[2].scatter(*self.__maxima.transpose(), c='r', edgecolors='face')
        self._plot_strokes(self.__strokes_width, "normalize width", axes[3])
        plt.show()

    @staticmethod
    def _plot_strokes(strokes, description, axes=None):
        """ plot current state of normalization

        Args:
            strokes (nested list of float[L,2]): (x,y)-points for each point in
                each stroke in each strokes
        """
        if axes is not None:
            axes.set_title(description)
            for stroke in strokes:
                axes.plot(stroke[:, 0], stroke[:, 1], 'g-')
            axes.set_aspect('equal')
            min_x, max_x, min_y, max_y = boundary_box(strokes)
            axes.plot([min_x, max_x], [0, 0], 'k:')
            axes.plot([min_x, max_x], [1, 1], 'k:')
            axes.set_xlim([min_x, max_x])
            axes.set_ylim([min_y, max_y])

    @staticmethod
    def normalize_skew(strokes):
        """ normalize skew by an linear fit

        Args:
            strokes (nested list of float[L,2]): (x,y)-points for each point in
                each stroke in each strokes
        """
        # TODO: only if long enough
        connected_strokes = np.concatenate(strokes)
        a, b = np.polyfit(connected_strokes[:, 0], connected_strokes[:, 1], 1)
        return [np.column_stack([stroke[:, 0],
                                 stroke[:, 1] - (a * stroke[:, 0] + b)])
                for stroke in strokes]

    @staticmethod
    def normalize_slant(strokes):
        # TODO(daniel): does nothing at the moment
        """ normalize the slant (not implemented yet)

        Args:
            strokes (nested list of float[L,2]): (x,y)-points for each point in
                each stroke in each strokes
        """
        # alphas = []
        # for stroke in strokes:
        #     delta = stroke[1:] - stroke[:-1]
        #     alphas += (np.arctan2(delta[:, 1], delta[:, 0]) % (np.pi)).tolist()
        # alphas = np.array(alphas)
        # alphas = (alphas + np.pi/2) % np.pi - np.pi/2
        # alphas = alphas[np.abs(alphas) < 0.2]
        return strokes
        # print(np.average(alphas))
        # plt.hist(alphas.tolist(),10)
        # plt.show()

    @staticmethod
    def normalize_baseline(strokes):
        """ normalize baseline to height zero and mean line to one

        Fits the local minima (maxima) to the baseline (mean line), resp.

        Args:
            strokes (nested list of float[L,2]): (x,y)-points for each point in
                each stroke in each strokes
        """

        minima = []
        maxima = []
        for stroke in strokes:
            # TODO(dv): avoid several consecutives extrema
            # (e.g. when maximum is divide)
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

        min_x, max_x, min_y, max_y = boundary_box(strokes)
        stroke_center = [np.average([min_x, max_x]),
                         np.average([min_y, max_y])]

        def get_normalization_func(minima, maxima):
            def fit_func(_, height): return height
            if np.average(maxima[:, 1]) > np.average(minima[:, 1]):
                # make only height detection since linear regression
                # is already done by normalize_skew
                fit_param_0 = np.array([0, 1])

                def error_func(fit_param, x1, y1, x2, y2):
                    return np.r_[
                        fit_func(x1, fit_param[0]) - y1,
                        fit_func(x2, fit_param[1]) - y2]
                fit_param, _ = optimize.leastsq(
                    error_func, fit_param_0,
                    args=(
                        minima[:, 0], minima[:, 1],
                        maxima[:, 0], maxima[:, 1]))
            else:
                fit_param = [stroke_center[1] - 0.5, stroke_center[1] + 0.5]
                # TODO(dv): change to standart height

            def rectify(points):
                height = fit_param[1]-fit_param[0]
                ret = np.zeros_like(points)
                ret[:, 0] = points[:, 0]/height
                ret[:, 1] = (points[:, 1] -
                             fit_func(points[:, 0], fit_param[0]))/height
                return ret
            return rectify

        rectify = get_normalization_func(minima, maxima)
        minima = minima[rectify(minima)[:, 1] < 1]
        maxima = maxima[rectify(maxima)[:, 1] > 0]
        rectify = get_normalization_func(minima, maxima)

        strokes = [rectify(s) for s in strokes]
        minima = rectify(minima)
        maxima = rectify(maxima)

        # def gaussian(x, x_0, sigma):
        #      return np.exp( - (x - x_0)**2 / (2 * sigma)**2 )
        # #def gaus_fit(x, x_0, sigma, )
        # gaus_param_0 = np.array([0, stroke_center[0]])
        # def error_func(param, minima, maxima):
        #     return  np.r_[
        #         gaussian(minima[:,0], *param) - minima[:,1],
        #         gaussian(maxima[:,0], *param) - maxima[:,1] - 1]
        # gaus_param, success = optimize.leastsq(
        #     error_func, gaus_param_0, args=(minima,maxima))
        # print(gaus_param)
        # def normalize_wih_gaussian(points):
        #     return (points
        #         - np.outer(gaussian(points[:,0], *gaus_param),np.array([0,1])))
        #
        # self.strokes = [normalize_wih_gaussian(s) for s in self.strokes]
        # minima = normalize_wih_gaussian(minima)
        # maxima = normalize_wih_gaussian(maxima)

        #
        # def smooth(points):
        #     ret = points.copy()
        #     for min_ in minima:
        #         print(min_)
        #         ret[:,1] += - max(0, min_[1])*gaussian(points[:,0], min_[0], 2)/1
        #     for max_ in maxima:
        #         print(min_)
        #         ret[:,1] += max(0,-max_[1]+1)*gaussian(points[:,0], max_[0], 2)/1
        #     return ret
        #
        # self.strokes = [smooth(s) for s in self.strokes]

        return strokes, minima, maxima

    @staticmethod
    def normalize_width(strokes):
        """ normalize the width by using the average width between two
            intersections with line between base and mean line

        Args:
            strokes (nested list of float[L,2]): (x,y)-points for each point in
                each stroke in each strokes
        """
        intersections = 0
        for stroke in strokes:
            is_below = stroke[:, 1] < 0.5
            is_below_merged = [x for x, _ in groupby(is_below)]
            intersections += len(is_below_merged) - 1

        min_x, max_x, _, _ = boundary_box(strokes)
        width = max_x - min_x
        scale_width = intersections / (2 * width) if width else 1
        return [np.column_stack([stroke[:, 0] * scale_width,
                                 stroke[:, 1]])
                for stroke in strokes]


class StrokeFeatures(NormalizeStroke):
    """ Generates features of a given stroke
    """

    NUM_FEATURES = 14
    DOTS_PER_UNIT = 5
    RDP_EPSILON = 0.02

    def __init__(self, strokes, normalize_first=True):
        """
        Args:
            strokes (list of float[L,2]): (x,y)-points for each point in
                each stroke in strokes
        """

        if normalize_first:
            super().__init__(strokes)
        else:
            self._strokes = strokes

        self._strokes = self._remove_dublicate_points(self._strokes)

    @staticmethod
    def _remove_dublicate_points(strokes):
        def remove_dublicate_from_stroke(stroke):
            return np.array([x for x, _ in groupby(stroke, tuple)])
        strokes = [remove_dublicate_from_stroke(stroke) for stroke in strokes]

        # connect strokes in the the consecutive stroke starts where the
        # proceding stroke ended:
        for i in list(range(len(strokes)-1))[::-1]:
            if all(strokes[i][-1] == strokes[i+1][0]):
                strokes[i:i+1] = np.concatenate([strokes[i][:-1],
                                                 strokes[i+1]])
        return strokes

    @lazyprop
    def _splines(self):
        return [self._create_spline(stroke) for stroke in self._strokes]

    @lazyprop
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
        for stroke in self._strokes:
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

    @lazyprop
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

    @lazyprop
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

    @lazyprop
    def _connected_stroke(self):
        """ x,y position of the splines (discretized)
        """
        stroke = []
        for spline, knots in zip(self._splines, self._knots):
            stroke += list(zip(*interpolate.splev(knots, spline)))
        return np.array(stroke)

    @lazyprop
    def _stroke_der(self):
        """ derivative of the spline
        """
        stroke_der = []
        for spline, knots in zip(self._splines, self._knots):
            stroke_der += list(zip(*interpolate.splev(knots, spline, der=1)))
        return np.array(stroke_der)

    @lazyprop
    def _stroke_cur(self):
        """ curvature of the spline
        """
        stroke_cur = []
        for i in range(len(self._splines)):
            stroke_cur += np.array(
                interpolate.splev(self._knots[i], self._splines[i],
                                  der=2)).transpose().tolist()
        return np.array(stroke_cur)

    @lazyprop
    def _tangent(self):
        """ normalized slope
        """
        return (self._stroke_der /
                np.sum(self._stroke_der**2, axis=1)[:, np.newaxis]**0.5)

    @lazyprop
    def radius_of_curvature(self):
        """ radius of a tangent circle at the point (positive and negaive)
        """
        with np.errstate(divide='ignore'):
            ret = (np.sum(self._stroke_der**2, axis=1)**(3/2) /
                   (self._stroke_der[:, 0] * self._stroke_cur[:, 1] -
                    self._stroke_der[:, 1] * self._stroke_cur[:, 0]))
        return ret

    @lazyprop
    def radius_normalized(self):
        """ a function of the radius
        """
        return (np.sign(self.radius_of_curvature) /
                (1 + np.abs(self.radius_of_curvature)))

    @lazyprop
    def delta_y(self):
        """ difference to the y-coordinate to the preceding point
        """
        return self._connected_stroke[1:, 1] - self._connected_stroke[:-1, 1]

    @lazyprop
    def delta_y_extended(self):
        """ difference to the y-coordinate to the preceding point and zero for
            the first point
        """
        return np.concatenate([[0], self.delta_y])

    @lazyprop
    def delta_x(self):
        """ difference to the x-coordinate to the preceding point
        """
        return self._connected_stroke[1:, 0]-self._connected_stroke[:-1, 0]

    @lazyprop
    def delta_x_extended(self):
        """ difference to the x-coordinate to the preceding point and zero for
            the first point
        """
        return np.concatenate([[0], self.delta_x])

    @lazyprop
    def pressure(self):
        """ Whether at the point the stroke starts, ends, or continues
        """
        cuts = [len(knots) for knots in self._knots]
        pressure = np.ones([sum(cuts), 2])
        for i in accumulate(cuts):
            pressure[i-1, 1] = 0
            pressure[i % len(pressure), 0] = 0
        return pressure

    @lazyprop
    def _angle_direction(self):
        """ The angle of the current writing and the horizontal line
        """
        alpha = np.arctan2(self._tangent[:, 1], self._tangent[:, 0])
        return alpha

    @lazyprop
    def writing_direction(self):
        """ The vector representation of _angle_direction
            (to avoid jump from 2*pi to 0)
        """
        return np.column_stack([np.cos(self._angle_direction),
                                np.sin(self._angle_direction)])

    @lazyprop
    def _angle_curvature(self):
        """ The difference of to consecutives angles giving the writing direction
        """
        tmp = np.zeros_like(self._angle_direction)
        tmp[1:-1] = self._angle_direction[2:]-self._angle_direction[1:-1]
        return tmp

    @lazyprop
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

    @lazyprop
    def x_low_pass_filtered(self):
        """ The deviation from the mean average movement in each stroke
        """
        return self.x_position - self.x_low_pass

    @lazyprop
    def curvature(self):
        """ The vector representation of _angle_curvature
            (to avoid jump from 2*pi to 0)
        """
        return np.sin(self._angle_curvature), np.cos(self._angle_curvature)

    @lazyprop
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

    @lazyprop
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

    @lazyprop
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

    @lazyprop
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
    ink_page, transcriptions = load("sample_text/the_zen_of_python.svg",
                                    "sample_text/the_zen_of_python.txt")
    ink_lines = split_lines(ink_page)
    ink = ink_lines[0]
    strokes_features = StrokeFeatures(ink, normalize_first=True)
    strokes_features.plot_normalization()
    strokes_features.plot_all()

if __name__ == "__main__":
    main()
