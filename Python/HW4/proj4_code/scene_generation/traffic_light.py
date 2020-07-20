import numpy as np
import cv2


class TrafficLight:

    def __init__(self, radius, state, name="traffic_light"):
        self.name = name
        self.radius = radius
        self.state = state
        self.image = None
        self.center = None
        self.dims = None

    def set_center(self, center):
        self.center = center

    def set_state(self, state):
        self.state = state
        self._create_traffic_light()

    def set_radius(self, radius):
        self.radius = radius

    def _create_traffic_light(self):
        offset = self.radius // 2
        w = 2 * self.radius + 2 * offset
        h = 6 * self.radius + 4 * offset

        self.center = {'x': offset + self.radius, 'y': 2 * offset + 3 * self.radius}
        self.dims = {'h': h, 'w': w}

        self.image = .2 * np.ones((h, w, 3))

        # Place lights
        if self.state == 'red':
            cv2.circle(self.image, (offset + self.radius, offset + self.radius), self.radius, (0, 0, 1), -1)
            cv2.circle(self.image, (offset + self.radius, 2 * offset + 3 * self.radius), self.radius, (0, .5, .5), -1)
            cv2.circle(self.image, (offset + self.radius, 3 * offset + 5 * self.radius), self.radius, (0, .5, 0), -1)

        elif self.state == 'yellow':
            cv2.circle(self.image, (offset + self.radius, offset + self.radius), self.radius, (0, 0, .5), -1)
            cv2.circle(self.image, (offset + self.radius, 2 * offset + 3 * self.radius), self.radius, (0, 1, 1), -1)
            cv2.circle(self.image, (offset + self.radius, 3 * offset + 5 * self.radius), self.radius, (0, .5, 0), -1)

        elif self.state == 'green':
            cv2.circle(self.image, (offset + self.radius, offset + self.radius), self.radius, (0, 0, .5), -1)
            cv2.circle(self.image, (offset + self.radius, 2 * offset + 3 * self.radius), self.radius, (0, .5, .5), -1)
            cv2.circle(self.image, (offset + self.radius, 3 * offset + 5 * self.radius), self.radius, (0, 1, 0), -1)

    def get_sign_image(self):
        self._create_traffic_light()

        return self.image

if __name__ == '__main__':
    test = TrafficLight(50, 'yellow')

    for state in ['red', 'yellow', 'green']:
        test.set_state(state)
        test_image = test.get_sign_image()
        cv2.imshow("Traffic Light", test_image)
        cv2.waitKey(1000)
