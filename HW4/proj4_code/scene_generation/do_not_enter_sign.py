import numpy as np
import cv2


class DoNotEnterSign:

    def __init__(self, height=None, name="do_not_enter_sign"):
        self.name = name
        self.height = height
        self.image = None
        self.dims = None

    def _create_sign(self):

        h = self.height
        self.image = np.zeros((h, h, 3)) - 1
        self.dims = {'h': self.image.shape[0], 'w': self.image.shape[1]}

        radius = int(h * 0.40)
        cv2.circle(self.image, (h // 2, h // 2), radius, (0, 0, 1), -1)

        rect_length = int(h * .65)
        rect_height = int(h * .15)
        rect_top_left = (h // 2 - rect_length // 2, h // 2 - rect_height // 2)
        rect_bot_right = (rect_top_left[0] + rect_length, rect_top_left[1] + rect_height)
        self.image[rect_top_left[1]:rect_bot_right[1], rect_top_left[0]:rect_bot_right[0]] = [1, 1, 1]

    def get_sign_image(self):
        self._create_sign()
        return self.image

if __name__ == '__main__':
    test = DoNotEnterSign(300)
    test_image = test.get_sign_image()
    cv2.imshow("Do Not Enter", test_image)
    cv2.waitKey(0)
