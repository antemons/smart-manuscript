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
import pylab as plt
from itertools import groupby, accumulate
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.optimize import minimize
from copy import deepcopy
from collections import OrderedDict
import warnings

from .utils import Transformation, cached_property

np.seterr(all="raise")
warnings.simplefilter('error', np.RankWarning)

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

__all__ = ["strokes_to_features", "NormalizationWarning"]


class Ink:
    """ Set of strokes

        nested list strokes (float[L,2]), the (x,y)-points for each point in
        each stroke in each strokes
    """

    def __init__(self, strokes=None, is_uncorrupted=False):
        """ create Ink from strokes, consider use Ink.from_corrupted_stroke()
        """
        #strokes = deepcopy(strokes)
        if strokes is not None and strokes != []:
            self._concatenated_strokes = np.concatenate(strokes)
            sections = np.cumsum([len(s) for s in strokes])[:-1]
            self.strokes = np.split(self._concatenated_strokes, sections)
        else:
            self._concatenated_strokes = np.array([])
            self.strokes = []
        self.is_uncorrupted = (is_uncorrupted or
            (hasattr(strokes, "is_uncorrupted") and strokes.is_uncorrupted))

    @classmethod
    def from_corrupted_stroke(cls, strokes, resort=False):
        """ Create Ink by removing dublicates points and bridges zero gaps

        >>> strokes = [np.array([[0, 0], [0, 0], [1, 0]]),\
                       np.array([[1, 0], [2, 0]]), np.array([[2, 1], [3, 1]])]
        >>> ink = Ink.from_corrupted_stroke(strokes)
        >>> print(ink)
        [[ 0.  0.]
         [ 1.  0.]
         [ 2.  0.]]
        [[ 2.  1.]
         [ 3.  1.]]
        """
        strokes = deepcopy(strokes)

        if resort:
            strokes = sorted(strokes, key=lambda x: np.mean(x[:, 0]))

        cls._connect_gapless_strokes(strokes)
        cls._remove_dublicated_points(strokes)

        return cls(strokes, is_uncorrupted=True)

    def __str__(self):
        return "".join(str(stroke) + "\n" for stroke in self)[:-1]

    def __getstate__(self):
        return [self.strokes, self.is_uncorrupted]

    def __setstate__(self, state):
        self.__init__(strokes=state[0], is_uncorrupted=state[1])

    def __iter__(self):
        return self.strokes.__iter__()

    @property
    def concatenated_strokes(self):
        return self._concatenated_strokes

    @staticmethod
    def _remove_dublicated_points(strokes):
        for i in range(len(strokes)):
            stroke = [point for point in strokes[i]]
            for j in reversed(range(len(stroke)-1)):
                if sum(abs(stroke[j] - stroke[j+1])) < 10**-7:
                    stroke[j] = (stroke[j] + stroke[j+1]) / 2
                    stroke.pop(j+1)
            strokes[i] = np.array(stroke)

    @staticmethod
    def _connect_gapless_strokes(strokes):
        """ connect strokes in the the consecutive stroke starts where the
            proceding stroke ended
        """
        for i in list(range(len(strokes)-1))[::-1]:
            if all(strokes[i][-1] == strokes[i+1][0]):
                strokes[i] = np.concatenate([strokes[i][:-1], strokes[i+1]])
                strokes.pop(i+1)

    @cached_property
    def boundary_box(self):
        return (min(self._concatenated_strokes[:, 0]),
                max(self._concatenated_strokes[:, 0]),
                min(self._concatenated_strokes[:, 1]),
                max(self._concatenated_strokes[:, 1]))

    @cached_property
    def width_height_ratio(self):
        min_x, max_x, min_y, max_y = self.boundary_box
        return (max_x - min_x) / (max_y - min_y)

    @cached_property
    def length(self):
        def stroke_length(stroke):
            x = stroke[:, 0]
            y = stroke[:, 1]
            dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
            return sum(dist)

        return sum(stroke_length(stroke) for stroke in self)

    @cached_property
    def part_length(self):
        """
        >>> strokes = [np.array([[0, 0], [1, 0]]),\
                       np.array([[3, 0], [4, 0], [4, 1]])]
        >>> ink = Ink(strokes)
        >>> print(ink.part_length)
        [ 0.  1.  3.  4.  5.]
        """
        deltas = self.concatenated_strokes[1:] - self.concatenated_strokes[:-1]
        distances = np.cumsum(np.sqrt(np.sum(deltas**2, axis=1)))
        return np.concatenate(([0], distances))

    def get_extrema(self):
        minima = []
        maxima = []
        for stroke in self:
            # TODO(dv): avoid several consecutives extrema
            #           (e.g. when maximum is divide)
            ys = stroke[:, 1]
            idx_min = argrelextrema(np.concatenate([ys, [np.inf]]),
                                    np.less_equal, mode='wrap')[0]
            idx_max = argrelextrema(np.concatenate([ys, [-np.inf]]),
                                    np.greater_equal, mode='wrap')[0]
            minima.extend(stroke[idx_min])
            maxima.extend(stroke[idx_max])
        minima, maxima = np.array(minima), np.array(maxima)
        return minima, maxima

    def plot_pylab(self, axes=None, transcription=None, auxiliary_lines=False,
                   hide_ticks=False):
        if axes is None:
            axes = plt.axes()
        if transcription is not None:
            axes.set_title(transcription)
        for stroke in self:
            axes.plot(stroke[:, 0], stroke[:, 1], 'k-')
        axes.set_aspect('equal')
        if hide_ticks:
            axes.get_xaxis().set_ticks([])
            axes.get_yaxis().set_ticks([])
        if auxiliary_lines:
            axes.plot(self.boundary_box[:2], [0, 0], 'k:')
            axes.plot(self.boundary_box[:2], [1, 1], 'k:')

    def __add__(self, other):
        return type(self).from_corrupted_stroke(
            self.strokes + other.strokes)

    def __rmatmul__(self, other):
        new_concatenated_strokes = other @ self.concatenated_strokes
        sections = np.cumsum([len(s) for s in self])[:-1]
        new_strokes = np.split(new_concatenated_strokes, sections)
        return type(self)(new_strokes, is_uncorrupted=self.is_uncorrupted)

# def gauss(x, mu, sigma):
#     return np.exp(- (x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
#
# class ExpectationMaximization:
#     """ The EM-Algorithm
#     """
#
#     def __init__(self, rel_frequency, sigma, initial_guess):
#         self._rel_frequency = rel_frequency
#         self._sigma = sigma
#         self._initial_guess = initial_guess
#
#     def estimation(self, xs, means):
#         p = np.array([[gauss(x, mu, self._sigma) * self._rel_frequency[i]
#                        for i, mu in enumerate(means)]
#                       for x in xs])
#         return p / np.sum(p, axis=1)[:, None]
#
#     def maximation(self, xs, means, E):
#         return np.sum(E * xs[:, None], axis=0) / np.sum(E, axis=0)
#
#     def __call__(self, xs):
#         means = self._initial_guess
#         while True:
#             estimation = self.estimation(xs, means)
#             old_means = means
#             means = self.maximation(xs, means, estimation)
#             if sum((old_means - means)**2) < 10**-9:
#                 break
#         return means
#
# get_baseline = ExpectationMaximization([.25, .25, .25, .25], 0.2,
#                                        [-1, 0, 1, 2])

class InkPage(Ink):
    """ Ink on a page """

    def __init__(self, strokes, page_size=None,
                 is_uncorrupted=False):
        super().__init__(strokes, is_uncorrupted=is_uncorrupted)
        self.page_size = page_size

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
        return [Ink.from_corrupted_stroke(line) for line in lines]

class NormalizationWarning(Warning):
    pass

warnings.simplefilter('always', NormalizationWarning)

class InkNormalization:
    """ normalize a stroke by slant, skew, baseline, height, and width
    """

    @staticmethod
    def _apply_normalizations(ink, normalizations, ret_steps=False):
        transformation = Transformation.identity()
        if ret_steps:
            steps = OrderedDict(original=(ink, transformation))
        for normalization in normalizations:
            ink, new_transformation = normalization(ink)
            transformation = new_transformation @ transformation
            if ret_steps:
                steps[normalization.__name__] = ink, transformation
        if ret_steps:
            return ink, transformation, steps
        return ink, transformation

    def __call__(self, ink, skew_is_horizontal=False):
        ink, transformation = self._apply_normalizations(
            ink, self._normaliztion_steps(skew_is_horizontal))
        return ink, transformation

    def _normaliztion_steps(self, skew_is_horizontal):
        if skew_is_horizontal:
            normalized_skew_and_mean = self.normalized_mean
        else:
            normalized_skew_and_mean = self.normalized_skew_and_mean
        return [normalized_skew_and_mean,
                self.normalized_skew_and_mean,
                self.normalized_slant,
                self.normalized_baseline,
                self.normalized_width,
                self.normalized_left]

    def plot(self, ink, skew_is_horizontal=False, axes=None):
        final_ink, final_transformation, steps = self._apply_normalizations(
            ink, self._normaliztion_steps(skew_is_horizontal), ret_steps=True)
        if axes is None:
            _, axes = plt.subplots(5)
        for axis, (name, (ink, transformation)) in zip(axes, steps.items()):
            ink.plot_pylab(
                axis, name,
                auxiliary_lines = name in ["normalized_baseline",
                                           "normalized_width"])

            if name == "original":
                width = final_ink.boundary_box[1]
                box = np.array([[0, 0], [width, 0], [width, 1], [0, 1]])
                axis.scatter(*((~final_transformation) @ box).transpose())

            if name == "normalized_skew_and_mean":
                axis.plot(ink.boundary_box[:2], [0, 0], 'k:')

            if name == "normalized_baseline":
                minima, maxima = ink.get_extrema()
                axis.scatter(*minima.transpose(), c='b', edgecolors='face')
                axis.scatter(*maxima.transpose(), c='r', edgecolors='face')
        return final_ink, final_transformation


    @staticmethod
    def normalized_mean(ink):
        mean = np.mean(ink.concatenated_strokes, axis=0)
        transformation = Transformation.translation(*(-mean))
        return transformation @ ink, transformation

    @classmethod
    def normalized_skew_and_mean(cls, ink):
        """ normalize skew by an linear fit

        Args:
            strokes (Ink): ink to normalize
        """
        part_length = ink.part_length

        try:
            a, b = np.polyfit(part_length, ink.concatenated_strokes, 1)
        except (ValueError, np.RankWarning, RuntimeWarning, FloatingPointError):
            # TODO(DV): single dot need special handling
            raise NormalizationWarning("Single dot is not Normalizeable")

        if np.linalg.norm(a) < 0.1:
            warnings.warn(NormalizationWarning(
                "Skew detection is ambigous ({})".format(np.linalg.norm(a))))

        angle = np.arctan2(*a[::-1])

        if np.linalg.norm(a) < 0.3:
            angle = round(angle/(np.pi/2))*np.pi/2
            transformation_skew = Transformation.rotation(-angle)
            _, transformation_mean = cls.normalized_mean(ink)
            #ink.transform(transformation)
            transformation = transformation_mean @ transformation_skew

        else:
            transformation = (Transformation.rotation(-angle) @
                              Transformation.translation(*(-b)))

        return transformation @ ink, transformation

    # def normalize_skew_old(self):
    #     """ normalize skew by an linear fit
    #
    #     Args:
    #         strokes (Ink): ink to normalize
    #     """
    #     # TODO: only if long enough
    #     #TODO(daniel): resampling in advance
    #
    #
    #     cov = np.cov(self.ink.concatenated_strokes.transpose())
    #     mean = np.mean(self.ink.concatenated_strokes, axis=0)
    #     eigvals, eigvecs = np.linalg.eig(cov)
    #     eigval, eigvec = sorted(zip(eigvals, eigvecs))[1]
    #     angle = -np.arctan2(*eigvec[::-1])
    #     self.has_been_well_normalized = max(eigvals) / min(eigvals) > 3
    #     print(eigvals)
    #     transformation = (Transformation.rotation(-angle) *
    #                       Transformation.translation(*(-mean)))
    #     self._transform(transformation)

    @staticmethod
    def normalized_left(ink):
        """ set the leftmost point to x = 0
        """
        min_x = ink.boundary_box[0]
        transformation = Transformation.translation(- min_x, 0)
        return transformation @ ink, transformation

    @staticmethod
    def normalized_slant(ink):
        """ normalize the slant
        """

        def tilted_ink_length(ink, angle):
            #ink = deepcopy(ink)
            #ink.transform(Transformation.shear(y_angle=angle))
            tilted_ink = Transformation.shear(y_angle=angle) @ ink
            return tilted_ink.length

        angle = minimize(lambda a: tilted_ink_length(ink, a), 0).x
        transformation = Transformation.shear(y_angle=angle)
        return transformation @ ink, transformation

    @staticmethod
    def normalized_baseline(ink):
        """ normalize baseline to y = 0 and mean line (height) to y = 1

        Fits the local minima (maxima) to the baseline (mean line), resp.

        Args:
            strokes (Ink): ink to normalize
        """

        minima, maxima = ink.get_extrema()
        try:
            min(minima[:, 1]) == max(maxima[:, 1])
        except IndexError:
            HEIGHT = 1
            BASELINE = 0
        if min(minima[:, 1]) == max(maxima[:, 1]):  # a horizontal line
                                                    # (close to y = 0)
            HEIGHT = 1
            BASELINE = 0
        else:
            minima = minima[minima[:, 1] <= 10**-12]
            maxima = maxima[maxima[:, 1] >= -10**-12]
            assert minima.size and maxima.size, (minima, maxima, ink)
            BASELINE = np.average(minima[:, 1])
            MEANLINE = np.average(maxima[:, 1])
            HEIGHT = MEANLINE - BASELINE
            assert HEIGHT >= 0
        transformation = (Transformation.scale(1 / HEIGHT) @
                          Transformation.translation(0, - BASELINE))
        transformed_ink = transformation @ ink

        if transformed_ink.boundary_box[2] < -4:
            warnings.warn(NormalizationWarning("Baseline normalization failed"))

        if transformed_ink.boundary_box[3] > 4:
            warnings.warn(NormalizationWarning("Baseline normalization failed"))

        return transformed_ink, transformation
    # def normalize_baseline_advanced(self):
    #     minima, maxima = self.ink.get_extrema()
    #     _, BASELINE, MEANLINE, _ = \
    #         get_baseline(np.concatenate([minima, maxima])[:, 1])
    #     HEIGHT = MEANLINE - BASELINE
    #     transformation = (Transformation.scale(1 / HEIGHT) *
    #                       Transformation.translation(0, - BASELINE))
    #     self._transform(transformation)

    @staticmethod
    def normalized_width(ink):
        """ normalize the width by using the average width between two
            intersections with line between base and mean line

        Args:
            strokes (Ink): ink to normalize
        """

        is_below = ink.concatenated_strokes[:, 1] < 0.5
        intersections = len([x for x, _ in groupby(is_below)])

        min_x, max_x, _, _ = ink.boundary_box
        width = max_x - min_x

        if (intersections >= 2) and width:
            scale_width = intersections / (4 * width)
        else:
            scale_width = 1
        transformation = Transformation.scale((scale_width, 1))
        return transformation @ ink, transformation

normalized = InkNormalization()







class InkFeatures:
    """ Generates features characterizing the ink
    """

    NUM_FEATURES = 15
    DOTS_PER_UNIT = 8
    # RDP_EPSILON = 0.02
    EPSILON = 10**-8

    def __init__(self, ink):
        """ Provide a list of features characterizing the set of strokes

        Args:
            strokes (Ink): the ink
        """
        if not ink.is_uncorrupted:
            raise ValueError("ink must be uncorrupted")
        self._ink = ink

    @classmethod
    def from_features(cls, features):
        ink_features = cls(Ink.from_corrupted_stroke([]))
        ink_features._cached_x_low_pass_filtered = features[:, 0]
        ink_features._cached_x_position = features[:, 1]
        ink_features._cached_y_position = features[:, 2]
        ink_features._cached_pressure = features[:, 3:5]
        ink_features._cached_radius_normalized = features[:, 5]
        ink_features._cached_writing_direction = features[:, 6:8]
        ink_features._cached__rough_tangent = features[:, 8:10]
        ink_features._cached_encased = features[:, 10:12]
        ink_features._cached_delta_x_extended = features[:, 12]
        ink_features._cached_delta_y_extended = features[:, 13]
        ink_features._cached_intersection = features[:, 14]
        return ink_features

    @classmethod
    def ink_to_features(cls, ink):
        return cls(ink).features

    @cached_property
    def _splines(self):
        return [self._create_spline(stroke) for stroke in self._ink]

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
        for stroke in self._ink:
            axes.plot(stroke[:, 0], stroke[:, 1], "bo")

    def plot_all(self, axes=None):
        """ plot the stroke
        """
        if axes is None:
            _, axes = plt.subplots(5, 1)
        for axis in axes:
            self._plot_splines(axis)
        self._plot_orig_strokes(axes[0])
        self._plot_encased(axes[1])
        self._plot_intersection(axes[2])
        self._plot_curvature(axes[3])
        self._plot_pressure(axes[4])

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
            dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
            print(dist)

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
                (self.EPSILON + np.sum(dpos_between**2, axis=1)**0.5)[:, np.newaxis]
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

    # def _get_knots_rdp(self, spline):
    #     # TODO: use shapely.geometry.simplify instead of rdp
    #     # this function is unsed (replaced by _get_knots_equidistant)
    #     from rdp import rdp
    #     u = np.linspace(0, spline[0][-1], max(2, 100*spline[0][-1]))
    #     stroke = np.array(interpolate.splev(u, spline)).transpose()
    #     intrpl_stroke = rdp(stroke, epsilon=self.RDP_EPSILON)
    #     intrpl_stroke_args = [
    #         (stroke[:, 0]+1j*stroke[:, 1]).tolist().index(dot[0]+1j*dot[1])
    #         for dot in intrpl_stroke]
    #     return u[intrpl_stroke_args]

    @cached_property
    def y_position(self):
        """ y-coordinate
        """
        return self._connected_stroke[:, 1]

    @cached_property
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
        return (self._stroke_der / (self.EPSILON +
            np.sum(self._stroke_der**2, axis=1))[:, np.newaxis]**0.5)

    @cached_property
    def radius_of_curvature(self):
        """ radius of a tangent circle at the point (positive and negaive)
        """
        with np.errstate(divide='ignore'):
            ret = (np.sum(self._stroke_der**2, axis=1)**(3/2) /
                   (self.EPSILON +
                    self._stroke_der[:, 0] * self._stroke_cur[:, 1] -
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
        for knots in self._knots:
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
        encased_from_top = 1*np.any((~ point_is_above)*tmp*conditions, axis=1)
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
            self.x_position,
            self.y_position,
            self.x_low_pass_filtered,
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

    FEATURES_NAMES = ["x_position",
                      "y_position",
                      "x_low_pass_filtered",
                      "pressure_start",
                      "pressure_stop",
                      "radius_normalized",
                      "writing_direction_x",
                      "writing_direction_y",
                      "rough_tangent_x",
                      "rough_tangent_y",
                      "encased_from_top",
                      "encased_from_below",
                      "delta_x_extended",
                      "delta_y_extended"]

def strokes_to_features(
        strokes,
        normalize=True,
        skew_is_horizontal=False,
        resort=False,
        ret_transformation=False):
    ink = Ink.from_corrupted_stroke(strokes, skew_is_horizontal)
    if normalize:
        ink, transformation = normalized(
            ink, skew_is_horizontal=skew_is_horizontal)
    features = InkFeatures.ink_to_features(ink)
    if not ret_transformation:
        return features
    else:
        return features, transformation

def plot_features(features, transcription=None, axes=None):
    """ show the strokes (only from the features)
    """
    x = features[:, 0]
    y = features[:, 1]

    if axes is None:
        axes = plt.axes()
    if transcription is not None:
        axes.set_title(transcription)
    axes.plot([min(x), max(x)], [0, 0], 'k:')
    axes.plot([min(x), max(x)], [1, 1], 'k:')
    axes.plot(x, y, "g-")
    axes.set_aspect('equal')
    # tmp = InkFeatures.from_features(features)
    # tmp.plot_all()
