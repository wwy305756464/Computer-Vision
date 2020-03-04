import numpy as np
import cv2


class YieldSign:

    def __init__(self, height=None, name="yield_sign"):
        self.name = name
        self.height = height
        self.outter_triangle_points = np.zeros((3, 2), dtype=np.int32)
        self.inner_triangle_points = np.zeros((3, 2), dtype=np.int32)
        self.image = None
        self.dims = None
        self.centroid = None

    def set_height(self, height):
        self.height = height

    def _setup(self):

        if self.height is None:
            raise ValueError("Height value is not defined")

        side_length = int(2. * self.height / 3**.5)

        self.outter_triangle_points[1, :] = self.outter_triangle_points[0] + \
                                            [side_length, 0]
        self.outter_triangle_points[2, :] = self.outter_triangle_points[0] + \
                                            [side_length // 2, self.height]

        self.centroid = np.array(np.rint(np.mean(self.outter_triangle_points,
                                                 axis=0)), np.int16)

        offset = self.height // 5
        sin_offset = offset * np.sin(np.pi / 6.)
        cos_offset = offset * np.cos(np.pi / 6.)
        self.inner_triangle_points[0, :] = self.outter_triangle_points[0] + \
                                           [cos_offset, sin_offset]
        self.inner_triangle_points[1, :] = self.outter_triangle_points[1] + \
                                           [-cos_offset, sin_offset]
        self.inner_triangle_points[2, :] = self.outter_triangle_points[2] + \
                                           [0, -offset]

    def _create_sign(self):
        self._setup()
        points = self.outter_triangle_points

        bottom_right = [int(np.max(self.outter_triangle_points[:, 1])) + 1,
                        int(np.max(self.outter_triangle_points[:, 0])) + 1]
        self.image = np.zeros((bottom_right[0], bottom_right[1], 3)) - 1

        self.dims = {'h': self.centroid[1] * 2, 'w': self.centroid[0] * 2}

        points = points.reshape((1, -1, 2))
        cv2.fillPoly(self.image, points, (0, 0, 1), 8)

        points = self.inner_triangle_points
        points = points.reshape((1, -1, 2))
        cv2.fillPoly(self.image, points, (1, 1, 1), 8)

    def get_sign_image(self):
        self._create_sign()
        return self.image

if __name__ == '__main__':
    test = YieldSign(300)
    test_image = test.get_sign_image()
    cv2.imshow("Yield Sign", test_image)
    cv2.waitKey(0)
