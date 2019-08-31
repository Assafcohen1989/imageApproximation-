import numpy as np
import cv2
from random import randrange, random, uniform


class Gene(object):
    def __init__(self, center=(0, 0), radius=0, color=None, opacity=randrange(100), frame_limit=(0, 0)):
        self._x, self._y = center[0], center[1]
        self._radius = radius
        self._color = color
        self._opacity = opacity
        self._frame_limit = frame_limit
        self.id = randrange(2000, 3000)

    def __repr__(self):
        return "center: ({0},{1}), radius: {2}, color: {3}, opacity: {4}%".format(self._x, self._y, self._radius, self._color, self._opacity)

    def generate(self, frame_limit=(0, 0)):
        if self._frame_limit == (0, 0):
            self._frame_limit = frame_limit
        self._radius = randrange(int((max(*self._frame_limit))/5))
        self._x = randrange(self._frame_limit[0])
        self._y = randrange(self._frame_limit[1])
        self._color = (randrange(255), randrange(255), randrange(255))
        self._opacity = randrange(50)
        return self

    def draw_gene(self, _img=None, use_opacity=True):
        if _img is None:
            _img = np.zeros((self._frame_limit[0], self._frame_limit[1], 3), np.uint8)

        if not use_opacity:
            cv2.circle(img=_img,
                       center=(self._x, self._y),
                       radius=self._radius,
                       color=self._color,
                       thickness=-1
                       )
        
        else:  # use_opacity = True
            overlay = np.zeros((self._frame_limit[0], self._frame_limit[1], 3), np.uint8)
            cv2.circle(img=overlay,
                       center=(self._x, self._y),
                       radius=self._radius,
                       color=self._color,
                       thickness=-1
                       )
            opacity = self._opacity/100
            cv2.addWeighted(_img, 1, overlay, opacity, 0, _img)

        show_weighted = False
        if show_weighted:
            # Show the results
            cv2.imshow('gene', _img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def copy(self):
        return Gene((self._x, self._y), self._radius, self._color, self._opacity, self._frame_limit)

    def mutate(self, num_of_mutations=1, step=0.15, p_radius=0.15, p_x=0.3, p_y=0.3, p_color=0.15, p_opacity=0.1):
        choices = np.random.choice(['radius', 'x', 'y', 'color', 'opacity'],
                                   num_of_mutations,
                                   p=[p_radius, p_x, p_y, p_color, p_opacity])
        change = uniform(-step, step)
        for choice in choices:
            for _ in range(5):  # 5 chances are give to mutate in the 'legal' range, otherwise mutation is skipped
                if choice == 'radius':
                    new_r = int(self._radius*change)
                    if 0 <= self._radius + new_r <= min(self._frame_limit[0], self._frame_limit[1]):
                        self._radius += int(self._radius*change)
                        break

                elif choice == 'x':
                    new_x = int(self._x*change)
                    if 0 <= self._x + new_x <= self._frame_limit[0]:
                        self._x += new_x
                        break

                elif choice == 'y':
                    new_y = int(self._y * change)
                    if 0 <= self._y + new_y <= self._frame_limit[1]:
                        self._y += new_y
                        break

                elif choice == 'color':
                    for channel in [0, 1, 2]:
                        new_color = int(self._color[channel]*change)
                        if new_color <= 0:
                            new_color = 0
                        elif new_color > 255:
                            new_color = 255
                        loc = list(self._color)
                        loc[channel] = new_color
                        self._color = tuple(loc)
                    break

                else:
                    new_opacity = int(self._opacity*change)
                    if 0 <= self._opacity + new_opacity <= 100:
                        self._opacity += new_opacity
                        break

