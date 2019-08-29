import numpy as np
import cv2
from random import randrange, random, uniform


class Gene(object):
    def __init__(self, center=(0, 0), radius=0, color=None, opacity=100, frame_limit=(0, 0)):
        self._x, self._y = center[0], center[1]
        self._radius = radius
        self._color = color
        self._opacity = opacity
        self._frame_limit = frame_limit
        self.id = randrange(2000, 3000)


    def __repr__(self):
        return "center: ({0},{1}), radius: {2}, color: {3}".format(self._x, self._y, self._radius, self._color)

    def generate(self, frame_limit=(0, 0), rgb=True):
        if self._frame_limit == (0, 0):
            self._frame_limit = frame_limit

        self._radius = randrange(min(*self._frame_limit))
        self._x = randrange(self._frame_limit[0])
        self._y = randrange(self._frame_limit[1])
        self._color = (randrange(255), randrange(255), randrange(255))
        self._opacity = randrange(100)
        return self

    def draw_gene(self, _img=None):
        if _img is None:
            _img = np.zeros((self._frame_limit[0], self._frame_limit[1], 3), np.uint8)
        #overlay = _img.copy()
        cv2.circle(img=_img,
                   center=(self._x, self._y),
                   radius=self._radius,
                   color=self._color,
                   thickness=-1
                   )
        ''' #Opacity is currently disabled
        if self._opacity != 100:
            opacity = self._opacity/100
            cv2.addWeighted(overlay, opacity, _img, 1 - opacity, 0, _img)
        '''

        ''' # Show the results
        cv2.imshow('gene', _img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

    def mutate(self, num_of_mutations=1, p_radius=0.2, p_x=0.2, p_y=0.2, p_color=0.2, p_opacity=0.2):
        choices = np.random.choice(['radius', 'x', 'y', 'color', 'opacity'],
                                   num_of_mutations,
                                   p=[p_radius, p_x, p_y, p_color, p_opacity])
        change = uniform(-0.15, 0.15)
        for choice in choices:

            if choice == 'radius':
                new_r = int(self._radius*change)
                if 0 <= self._radius + new_r <= min(self._frame_limit[0], self._frame_limit[1]):
                    self._radius += int(self._radius*change)

            elif choice == 'x':
                new_x = int(self._x*change)
                if  0 <= self._x + new_x <= self._frame_limit[0]:
                    self._x += new_x

            elif choice == 'y':
                new_y = int(self._y * change)
                if 0 <= self._y + new_y <= self._frame_limit[1]:
                    self._y += new_y

            elif choice == 'color':
                channel = randrange(3)
                new_color = int(self._color[channel]*change)
                if 0 <= new_color <= 255:
                    loc = list(self._color)
                    loc[channel] = new_color
                    self._color = tuple(loc)

            else:
                new_opacity = int(self._opacity*change)
                if 0 <= self._opacity + new_opacity <= 100:
                    self._opacity += new_opacity

