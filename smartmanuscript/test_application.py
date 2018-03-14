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
import os
from collections import namedtuple

from . import application


def test_application():
    Event = namedtuple("Event", ["x", "y"]) 
    class TestHandwrittenInput(application.HandwrittenInput):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.on_button_press(None, event=Event(150, 150))
            self.on_mouse_move(None, event=Event(175, 125))
            self.on_mouse_move(None, event=Event(200, 100))
            self.on_mouse_move(None, event=Event(200, 125))
            self.on_mouse_move(None, event=Event(200, 150))
            self.on_mouse_move(None, event=Event(200, 175))
            self.on_mouse_move(None, event=Event(200, 200))
            self.on_button_release()
    model_path = os.path.join(
        os.path.dirname(__file__), 'data', 'model', 'model.ckpt')
    app = TestHandwrittenInput(model_path)
    assert app.buttons[0].get_label() == "1", \
        "wrong recognition or error while window opening"
    app.quit()

