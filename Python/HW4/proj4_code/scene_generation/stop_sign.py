import numpy as np
import cv2


class StopSign:

    def __init__(self, height, name="stop_sign"):
        self.name = name
        self.height = height
        self.side_length = int(height / (1. + 2**.5))
        self.oct_points = np.zeros((8, 2), dtype=np.int32)
        self.image = None
        self.dims = None

    def _setup(self):
        r = int(100 / (1. + 2**.5))
        self.oct_points[1, :] = self.oct_points[0, :] + [r, 0]
        self.oct_points[2, :] = self.oct_points[1, :] + [r * np.cos(np.pi / 4), r * np.sin(np.pi / 4)]
        self.oct_points[3, :] = self.oct_points[2, :] + [0, r]
        self.oct_points[4, :] = self.oct_points[3, :] + [-r * np.cos(np.pi / 4), r * np.sin(np.pi / 4)]
        self.oct_points[5, :] = self.oct_points[4, :] + [-r, 0]
        self.oct_points[6, :] = self.oct_points[5, :] + [-r * np.cos(np.pi / 4), -r * np.sin(np.pi / 4)]
        self.oct_points[7, :] = self.oct_points[6, :] - [0, r]

        self.oct_points[:, 0] += abs(min(np.min(self.oct_points[:, 0]), 0))
        self.oct_points[:, 1] += abs(min(np.min(self.oct_points[:, 1]), 0))

    def _create_outline(self):

        self._setup()

        bottom_right = [int(np.max(self.oct_points[:, 1])) + 1, int(np.max(self.oct_points[:, 0])) + 1]

        self.image = np.zeros((bottom_right[0], bottom_right[1], 3))

        for i in range(8):
            pt1 = (int(self.oct_points[i, 0]), int(self.oct_points[i, 1]))

            if i == 7:
                pt2 = (int(self.oct_points[0, 0]), int(self.oct_points[0, 1]))
            else:
                pt2 = (int(self.oct_points[i + 1, 0]), int(self.oct_points[i + 1, 1]))

            cv2.line(self.image, pt1, pt2, (1, 1, 1))

    def _create_sign(self):
        self._setup()
        points = self.oct_points

        bottom_right = [int(np.max(self.oct_points[:, 1])) + 1, int(np.max(self.oct_points[:, 0])) + 1]
        self.image = np.zeros((bottom_right[0], bottom_right[1], 3)) - 1

        points = points.reshape((1, -1, 2))
        cv2.fillPoly(self.image, points, (0, 0, 204. / 255), 8)

        text_height = int(100 / (1. + 2**.5))
        lower_left = (self.oct_points[0][0] - text_height // 2 + 2, self.oct_points[6][1] - text_height // 4)
        cv2.putText(self.image, 'STOP', lower_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 2)

    def get_sign_image(self):
        self._create_sign()

        if self.height != 100:
            output_img = cv2.resize(self.image, (self.height, self.height), interpolation=cv2.INTER_CUBIC)
        else:
            output_img = self.image.copy()

        self.dims = {'h': output_img.shape[0], 'w': output_img.shape[1]}

        return output_img

if __name__ == '__main__':
    test = StopSign(200)
    test_image = test.get_sign_image()

    noise = np.random.normal(0, .4, (test_image.shape[0], test_image.shape[1], 3))
    test_image[:, :, :] += noise

    cv2.imshow("Stop Sign", test_image)

    test_image = np.clip(255 * test_image, 0, 255)
    cv2.imwrite("output_images/stop_sign.png", test_image)

    cv2.waitKey(0)
