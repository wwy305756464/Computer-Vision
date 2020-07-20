import numpy as np
import cv2

from scene_generation.traffic_light import TrafficLight
from scene_generation.do_not_enter_sign import DoNotEnterSign
from scene_generation.stop_sign import StopSign
from scene_generation.yield_sign import YieldSign
from scene_generation.warning_sign import WarningSign


class Scene:

    def __init__(self, dimensions=None):
        self.dims = dimensions
        self.image = None
        self.objects = {}

    def _add_object_to_dict(self, label, x, y, dims):
        self.objects[label] = {"x": x, "y": y, "dims": dims}

    def _create_background(self, dimensions=None):

        if dimensions is not None:
            self.dims = dimensions

        h, w = self.dims
        self.image = .5 * np.ones((h, w, 3))

        # Render sky
        self.image[:h // 2, :, :] = (1., 178. / 255, 102. / 255)

        # Render grass
        self.image[h // 2:3 * h // 4, :, :] = (0, 204. / 255, 0)

        # Place sun
        sun_radius = h // 15
        x_sun = np.random.randint(sun_radius, w - sun_radius)
        y_sun = np.random.randint(sun_radius, h // 2 - sun_radius)
        cv2.circle(self.image, (x_sun, y_sun), sun_radius, (0, 1, 1), -1)

        self._add_object_to_dict("sun", x_sun, y_sun, {'h': sun_radius, 'w': sun_radius})

    def place_traffic_light(self, top_left, radius, state):
        self._create_background()

        tf = TrafficLight(radius, state)
        traffic_light_image = tf.get_traffic_light_image()

        h, w = self.dims
        top_left = (min(top_left[1], w - tf.dims['w']), min(top_left[0], h - tf.dims['h']))

        self.image[top_left[1]:top_left[1] + tf.dims['h'],
                   top_left[0]:top_left[0] + tf.dims['w']] = traffic_light_image

    def place_sign(self, top_left, sign_obj):

        if self.image is None:
            self._create_background()

        sign_image = sign_obj.get_sign_image()

        h, w = self.dims
        top_left = (min(top_left[1], h - sign_obj.dims['h']), min(top_left[0], w - sign_obj.dims['w']))

        valid = sign_image >= 0

        boolean_image = np.zeros(self.image.shape, dtype=bool)
        boolean_image[top_left[0]:top_left[0] + valid.shape[0],
                      top_left[1]:top_left[1] + valid.shape[1], :] = valid[:, :, :]

        self.image[boolean_image] = sign_image[valid]

        x_obj = top_left[1] + sign_obj.dims['w'] // 2
        y_obj = top_left[0] + sign_obj.dims['h'] // 2
        self._add_object_to_dict(sign_obj.name, x_obj, y_obj, sign_obj.dims)

    def get_scene(self):

        if self.image is None:
            self._create_background()
            return self.image

        else:
            return self.image

    def get_objects(self):
        return self.objects

    def print_objects_data(self):

        print("Data of each object in the scene:\n")

        for label in self.objects:

            print(label)

            for k in self.objects[label]:
                print( "{}: {}".format(k, self.objects[label][k]))

            print("") 

class BlankScene(Scene):
    def __init__(self, dimensions=None, backgroundColor=(255,255,255)):
        self.dims = dimensions
        self.image = None
        self.objects = {}
        self.backgroundColor = backgroundColor

    def _create_background(self, dimensions=None):
        if dimensions is not None:
            self.dims = dimensions

        h, w = self.dims
        self.image = np.empty((h, w, 3))
        #make background of a homogenous color
        self.image[:,:] = np.array(list(map(lambda color: color/255, self.backgroundColor)))

if __name__ == '__main__':
    test = Scene((600, 1000))
    # test.place_traffic_light((100, 100), 10, 'yellow')

    tf = TrafficLight(10, 'red')
    test.place_sign((100, 300), tf)

    dne = DoNotEnterSign(70)
    test.place_sign((200, 300), dne)

    stp = StopSign(100)
    test.place_sign((300, 300), stp)

    yld = YieldSign(100)
    test.place_sign((450, 300), yld)

    work = WarningSign(100, (0, 128. / 255, 1))
    test.place_sign((600, 300), work)

    wrng = WarningSign(100)
    test.place_sign((750, 300), wrng)

    test_image = test.get_scene()

    test.print_objects_data()

    noise = np.random.normal(0, .4, (test_image.shape[0], test_image.shape[1], 3))
    test_image[:, :, :] += noise

    cv2.imshow("Scene", test_image)
    test_image = np.clip(255 * test_image, 0, 255)
    cv2.imwrite("output_images/scene_all_signs.png", test_image)

    cv2.waitKey(0)
