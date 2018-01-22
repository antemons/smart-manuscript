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

import xml.etree.ElementTree as ET
import re
import numpy as np
import pylab as plt
from functools import reduce

from .utils import cached_property

__author__ = "Daniel Vorberg"
__copyright__ = "Copyright (c) 2017, Daniel Vorberg"
__license__ = "GPL"

__all__ = ["InkML"]


class Element:

    @classmethod
    def from_xml_node(cls, node):
        """ Create new element from XML

        Args:
            node (xml.etree.ElementTree.Element): data to read
        """
        ELEMENT_TYPES = {
            "trace": Trace,
            "annotationXML": AnnotationXML,
            "annotation": Annotation,
            "traceGroup": TraceGroup,
            "traceView": TraceView}
        element_type = ELEMENT_TYPES.get(node.tag, Element)
        return element_type(node)

    def __init__(self, node):
        self._node = node

    @cached_property
    def childs(self):
        return [Element.from_xml_node(child) for child in self._node]

    def __iter__(self):
        yield from self.childs

    def childs_of_type(self, types):
        if type(types) is list:
            yield from filter(lambda c: type(c) in types, self)
        else:
            yield from filter(lambda c: type(c) is types, self)

    @cached_property
    def _trace(self):
        return reduce(lambda a, b: dict(a, **b),
                      (child._trace for child in self), {})

    def ink(self, trace_refs=None):
        if trace_refs is None:
            return list(trace for _, trace in sorted(self._trace.items()))
        else:
            return [self._trace[trace_ref.replace("#", "")]
                    for trace_ref in trace_refs]

    @property
    def annotation(self):
        annotations = self.childs_of_type([Annotation, AnnotationXML])
        return reduce(lambda a, b: dict(a, **b),
                      (annotation.get() for annotation in annotations), {})

    def search(self, condition):
        if condition(self):
            yield self
        for child in self:
            yield from child.search(condition)


class TraceGroup(Element):
    pass


class Annotation(Element):

    def get(self):
        return {self._node.attrib["type"]: self._node.text}


class AnnotationXML(Element):

    def get(self):
        result = {}
        for child in self._node:
            result[child.tag] = child.text
        return result


class Trace(Element):
    new_unique_id = 0
    @property
    def _trace(self):
        return {self.id(): self.stroke()}

    def id(self):
        for key, value in self._node.attrib.items():
            if key[-2:] == "id":
                return value
        Trace.new_unique_id += 1
        return str(Trace.new_unique_id)

    def stroke(self):
        trace = self._node.text
        return parse(trace)


class TraceView(Element):

    def _trace_data_refs(self):
        trace_data_ref = self._node.get("traceDataRef")
        if trace_data_ref is not None:
            yield trace_data_ref
        for trace_view in self.childs_of_type(TraceView):
            yield from trace_view._trace_data_refs()


class InkML:
    """ Parser for InkML files

    Documentation: https://www.w3.org/2003/InkML
    """

    def __init__(self, filename):

        with open(filename) as file_:
            xmlstring = file_.read()
        xmlstring = re.sub(' xmlns="[^"]+"', '', xmlstring, count=1)
        self.filename = filename
        self.xml_root = ET.fromstring(xmlstring)
        self._root = Element.from_xml_node(self.xml_root)

    def ink(self, trace_refs=None):
        return self._root.ink(trace_refs)

    def plot(self, axes=None, transcription=None,
             hide_ticks=False):
        if axes is None:
            axes = plt.axes()
        if transcription is not None:
            axes.set_title(transcription)
        for stroke in self.ink():
            axes.plot(stroke[:, 0], stroke[:, 1], 'k-')
        axes.set_aspect('equal')
        if hide_ticks:
            axes.get_xaxis().set_ticks([])
            axes.get_yaxis().set_ticks([])


class Channel:

    PREFIX_EXPLICIT = "!"
    PREFIX_VELOCITY = "'"
    PREFIX_ACCELERATION = '"'
    PREFIXES = [PREFIX_EXPLICIT, PREFIX_VELOCITY, PREFIX_ACCELERATION]

    def __init__(self):
        self._data = []
        self._prefix = self.PREFIX_EXPLICIT
        self._last_velocity = None

    def get_data(self):
        return self._data

    @property
    def _last_coordinate(self):
        if len(self._data):
            return self._data[-1]
        else:
            return None

    def append(self, value):
        if value == "T":
            self._data.append(True)
        elif value == "F":
            self._data.append(False)
        elif value in ["*", "?"]:
            self._data.append(value)
        else:
            if value[0] in self.PREFIXES:
                self._prefix = value[0]
                value = value[1:]
            if self._prefix == self.PREFIX_EXPLICIT:
                coordinate = float(value)
            elif self._prefix == self.PREFIX_VELOCITY:
                assert self._last_coordinate is not None
                velocity = float(value)
                coordinate = self._last_coordinate + velocity
                self._last_velocity = velocity
            elif self._prefix == self.PREFIX_ACCELERATION:
                assert self._last_coordinate is not None
                assert self._last_velocity is not None
                acceleration = float(value)
                velocity = self._last_velocity + acceleration
                coordinate = self._last_coordinate + velocity
                self._last_velocity = velocity
            self._data.append(coordinate)


class TraceParser:
    """ Parse the trace of an InkML file

    Definition: https://www.w3.org/TR/InkML/#traces

    >>> parse = TraceParser()
    >>> trace ="10 0, 9 14"
    >>> print(parse(trace))
    [[ 10.   0.]
     [  9.  14.]]
    """

    def __call__(self, trace):
        points = trace.split(",")
        # TODO(daniel): check if last point is valid
        channel_x = Channel()
        channel_y = Channel()
        for point in points:
            number = r"([+-]\s*)?([0-9]*[.])?[0-9]+"
            value = r"(\"|'|!)?\s*{}".format(number)
            rule = r"(?P<x>{0})(?P<y>{0}).*".format(value)
            match = re.match(rule, point)
            channel_x.append(match.group('x'))
            channel_y.append(match.group('y'))

        stroke = np.array([channel_x.get_data(),
                           channel_y.get_data()]).transpose()
        return stroke

parse = TraceParser()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='InkML Reader')
    parser.add_argument('file', default="test.inkml", type=str, nargs='?')
    args = parser.parse_args()
    inkml = InkML(args.file)
    inkml.plot()
    plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
