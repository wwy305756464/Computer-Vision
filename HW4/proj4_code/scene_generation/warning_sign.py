import numpy as np
import cv2


class WarningSign:

    def __init__(self, height=None, color=(0, 1, 1), name="warning_sign"):
        self.name = name
        self.height = height
        self.color = color
        self.outter_diamond_points = np.zeros((4, 2), dtype=np.int32)
        self.inner_diamond_points_1 = np.zeros((4, 2), dtype=np.int32)
        self.inner_diamond_points_2 = np.zeros((4, 2), dtype=np.int32)
        self.image = None
        self.dims = None

    def set_height(self, height):
        self.height = height

    def _setup(self):

        if self.height is None:
            raise ValueError("Height value is not defined")

        h = self.height

        self.outter_diamond_points[0, :] = [h // 2, 0]
        self.outter_diamond_points[1, :] = [h, h // 2]
        self.outter_diamond_points[2, :] = [h // 2, h]
        self.outter_diamond_points[3, :] = [0, h // 2]

        self.inner_diamond_points_1[0] = self.outter_diamond_points[0] + [0, h // 50]
        self.inner_diamond_points_1[1] = self.outter_diamond_points[1] + [- h // 50, 0]
        self.inner_diamond_points_1[2] = self.outter_diamond_points[2] + [0, -h // 50]
        self.inner_diamond_points_1[3] = self.outter_diamond_points[3] + [h // 50, 0]

        self.inner_diamond_points_2[0] = self.inner_diamond_points_1[0] + [0, h // 50]
        self.inner_diamond_points_2[1] = self.inner_diamond_points_1[1] + [- h // 50, 0]
        self.inner_diamond_points_2[2] = self.inner_diamond_points_1[2] + [0, -h // 50]
        self.inner_diamond_points_2[3] = self.inner_diamond_points_1[3] + [h // 50, 0]

    def _create_sign(self):
        self._setup()
        points = self.outter_diamond_points

        bottom_right = [self.height, self.height]
        self.image = np.zeros((bottom_right[0], bottom_right[1], 3)) - 1

        self.dims = {'h': self.image.shape[0], 'w': self.image.shape[1]}

        points = points.reshape((1, -1, 2))
        cv2.fillPoly(self.image, points, self.color, 8)

        points = self.inner_diamond_points_1

        points = points.reshape((1, -1, 2))
        cv2.fillPoly(self.image, points, (0, 0, 0), 8)

        points = self.inner_diamond_points_2

        points = points.reshape((1, -1, 2))
        cv2.fillPoly(self.image, points, self.color, 8)

    def get_sign_image(self):
        self._create_sign()
        return self.image

class ConstuctionSign(WarningSign):
    def __init__(self, height=None, color=(0, 128./255, 1),
                 name="construction_sign"):
        self.name = name
        self.height = height
        self.color = color
        self.outter_diamond_points = np.zeros((4, 2), dtype=np.int32)
        self.inner_diamond_points_1 = np.zeros((4, 2), dtype=np.int32)
        self.inner_diamond_points_2 = np.zeros((4, 2), dtype=np.int32)
        self.image = None
        self.dims = None
        
if __name__ == '__main__':
    test = WarningSign(300, (0, 128. / 255, 1))
    test_image = test.get_sign_image()
    cv2.imshow("Warning Sign", test_image)
    cv2.waitKey(0)
