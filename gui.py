""" Utilities for GUIs """

# Imports
from collections import namedtuple

import numpy as np

# Constants

SelRect = namedtuple('SelRect', 'sel_class, x0, y0, x1, y1')
SelPoint = namedtuple('SelPoint', 'sel_class, x, y')


# Classes


class ActionStack(object):
    """ Stack for all the actions in the undo/redo queue """

    def __init__(self):
        self.stack = []
        self.stack_idx = 0

    def do(self, action, state):
        """ Do an action and update the state """

        stack = self.stack[:self.stack_idx]
        action.do(state)

        self.stack = stack
        self.stack.append(action)
        self.stack_idx = len(self.stack)

    def undo(self, state):
        """ Undo the last action, updating the state """

        if self.stack_idx > 0:
            action = self.stack[self.stack_idx - 1]
            action.undo(state)
            self.stack_idx -= 1

    def redo(self, state):
        """ Redo the last action, updating the state """

        if self.stack_idx < len(self.stack):
            action = self.stack[self.stack_idx]
            action.do(state)
            self.stack_idx += 1

    def clear(self):
        """ Clear the action queue """
        self.stack = []
        self.stack_idx = 0

    def __repr__(self):
        actions = []

        for i, action in enumerate(self.stack):
            actions.append(repr(action))
            if i == self.stack_idx - 1:
                actions.append('*')
        return 'ActionStack(' + ', '.join(actions) + ')'


class Action(object):
    """ Base class for undo/redo support """

    def do(self, state):
        raise NotImplementedError('Implement do')

    def undo(self, state):
        raise NotImplementedError('Implement undo')

    def __repr__(self):
        name = self.__class__.__name__
        keys = []
        for key, val in self.__dict__.items():
            if key.startswith('_'):
                continue
            keys.append('{}={}'.format(key, val))
        return '{}({})'.format(name, ', '.join(keys))

    __str__ = __repr__


# Concrete actions


class DrawShapeAction(Action):
    """ Action for drawing a shape on the GUI """

    def __init__(self, shape):
        self.shape = shape

    def do(self, state):
        """ Draw a shape on the state """
        state.add_shape(self.shape)

    def undo(self, state):
        """ Delete a shape from the state """
        state.remove_shape(self.shape)


class ClearAllAction(Action):
    """ Action for clearing the view """

    def __init__(self, shapes):
        self.shapes = set(shapes)

    def do(self, state):
        for shape in self.shapes:
            state.remove_shape(shape)

    def undo(self, state):
        for shape in self.shapes:
            state.add_shape(shape)


# Main tagging GUI


class ShapeManager(object):
    """ Manage shapes that we draw on images """

    def __init__(self, window=None):

        # Handle to the window
        self.window = window

        # Action queue for shapes
        self.action_stack = ActionStack()

        # Bounding coordinates for the axis
        self.x0, self.x1 = None, None
        self.y0, self.y1 = None, None
        self.xlen, self.ylen = None, None

        # Shapes
        self.points = set()
        self.rectangles = set()

    def find_near_points(self, x, y, radius=1.0):
        """ Find points near a center point """
        if not self.points:
            return []
        points = list(self.points)
        point_x = np.array([p.x - x for p in points])
        point_y = np.array([p.y - y for p in points])
        dist = (point_x**2 + point_y**2) <= radius**2
        return [p for p, i in zip(points, dist) if i]

    def add_shape(self, shape):
        """ Add a shape to the GUI """
        if isinstance(shape, SelRect):
            self.rectangles.add(shape)

            if self.window is not None:
                self.window.draw_rectangle(shape)
        elif isinstance(shape, SelPoint):
            self.points.add(shape)

            if self.window is not None:
                self.window.draw_point(shape)
        else:
            raise TypeError('Unknown shape: {}'.format(shape))

    def remove_shape(self, shape):
        """ Remove a shape from the GUI """

        if isinstance(shape, SelRect):
            if shape in self.rectangles:
                self.rectangles.remove(shape)
            if self.window is not None:
                self.window.remove_rectangle(shape)

        elif isinstance(shape, SelPoint):

            if shape in self.points:
                self.points.remove(shape)

            if self.window is not None:
                self.window.remove_point(shape)
        else:
            raise TypeError('Unknown shape: {}'.format(shape))

    def load_axis_bounds(self, bbox, aspect):
        """ Load the boundaries for the axis """

        (ix0, iy0), (ix1, iy1) = bbox
        if ix0 > ix1:
            ix0, ix1 = ix1, ix0
        if iy0 > iy1:
            iy0, iy1 = iy1, iy0

        ixc = (ix0 + ix1)/2
        iyc = (iy0 + iy1)/2

        ixlen = ix1 - ix0
        iylen = iy1 - iy0

        axis_aspect = iylen / ixlen

        if axis_aspect > aspect:
            # x too big
            ixlen = iylen / aspect
            ix1 = ixc + ixlen/2
            ix0 = ixc - ixlen/2
        else:
            # y too big
            iylen = ixlen * aspect
            iy1 = iyc + iylen/2
            iy0 = iyc - iylen/2

        self.x0, self.x1 = ix0, ix1
        self.y0, self.y1 = iy0, iy1
        self.xlen = ix1 - ix0
        self.ylen = iy1 - iy0

    def warp_to_figure(self, *args):
        """ Warp from axis coordinates to figure coordinates

        :param *args:
            Any number of x, y pairs
        :returns:
            The transformed x, y pairs
        """
        assert len(args) % 2 == 0

        warp_args = []
        for x, y in zip(args[::2], args[1::2]):
            fx = x * self.xlen + self.x0
            fy = y * self.ylen + self.y0
            warp_args.extend([fx, fy])
        return warp_args

    def warp_to_axis(self, *args):
        """ Warp from figure coordinates to axis coordinates """

        assert len(args) % 2 == 0

        warp_args = []
        for x, y in zip(args[::2], args[1::2]):

            # Clamp to axis bounds
            fx = max([self.x0, min([self.x1, x])])
            fy = max([self.y0, min([self.y1, y])])

            px = (fx - self.x0) / self.xlen
            py = (fy - self.y0) / self.ylen

            warp_args.extend([px, py])
        return warp_args

    def on_rectangle_complete(self, sel_class, x0, y0, x1, y1):
        """ Callback for when we finish a rectangle """
        rect = SelRect(sel_class, x0, y0, x1, y1)
        self.action_stack.do(DrawShapeAction(rect), self)

    def on_point_complete(self, sel_class, x, y):
        """ Callback for when we finish a point """
        point = SelPoint(sel_class, x, y)
        self.action_stack.do(DrawShapeAction(point), self)

    def on_clear_all(self):
        """ Called whenever objects get cleared """
        shapes = self.rectangles | self.points
        self.action_stack.do(ClearAllAction(shapes), self)

    def on_window_resize(self):
        """ Called whenever the window size changes """
        if self.window is None:
            return

        shapes = self.rectangles | self.points
        for shape in shapes:
            self.remove_shape(shape)

        # Update the window
        self.window.load_bounds()

        for shape in shapes:
            self.add_shape(shape)

    def on_undo(self):
        self.action_stack.undo(self)

    def on_redo(self):
        self.action_stack.redo(self)

    def on_reset_actions(self):
        self.action_stack.clear()
