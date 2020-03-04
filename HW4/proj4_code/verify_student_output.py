import unittest

import numpy as np


class OutputCheck(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check_nonetype(self, output):
        self.assertTrue(output is not None, "Output is NoneType. It seems this function has not "
                                            "been implemented yet or does not have a return call.")

    def check_ndarray(self, output, ndim=None, shape=None, dtype=None, msg=''):
        is_np_array = isinstance(output, np.ndarray)
        self.assertTrue(is_np_array, "Output is not a numpy.array. "
                                     "Your function returned an object of {}. "
                                     "{}".format(type(output), msg))

        empty = np.any(np.array(list(output.shape)) == 0)
        self.assertTrue(not empty, "At least one dimension of the array is empty.\n"
                                   "Student's output array shape: {}. {}".format(output.shape, msg))

        is_matrix = isinstance(output, np.matrix)
        self.assertTrue(not is_matrix, "Output returned a matrix and not a numpy.array. "
                                       "Your function returned an object of {}. "
                                       "{}".format(type(output), msg))

        if ndim is not None:
            output_ndim = len(output.shape)
            is_correct_ndim = ndim == output_ndim
            self.assertTrue(is_correct_ndim, "Output must be a {}D array. Your function returned an array of {} "
                                             "dimensions. "
                                             "{}".format(ndim, output_ndim, msg))
        if shape is not None:
            output_shape = output.shape
            is_correct_shape = np.array_equal(shape, output_shape)
            self.assertTrue(is_correct_shape, "Output must be an array with shape {}. Your function returned "
                                              "an array with shape {}. "
                                              "{}".format(shape, output_shape, msg))

        if dtype is not None:
            output_dtype = output.dtype
            is_correct_dtype = dtype == output_dtype
            self.assertTrue(is_correct_dtype, "Output must be a numpy.array with dtype '{}'. Your function returned "
                                              "an array with dtype '{}'. "
                                              "{}".format(dtype, output_dtype, msg))

    def check_np_array(self, output):
        is_np_array = isinstance(output, (np.ndarray))
        self.assertTrue(is_np_array, "Output is not a numpy.array. "
                                     "Your function returned an object of {}".format(type(output)))

    def check_array_dims(self, output, array_type):
        dims = output.shape

        if array_type == "2d":
            self.assertTrue(len(dims) == 2, "Output must be a 2D array. Your function returned an array of {} "
                                            "dimensions.".format(len(dims)))

        elif array_type == "3d":
            self.assertTrue(len(dims) == 3, "Output must be a 3D array. Your function returned an array of {} "
                                            "dimensions.".format(len(dims)))

    def check_tuple(self, output, length):
        is_tuple = isinstance(output, (tuple))
        self.assertTrue(is_tuple, "Output is not a tuple. "
                                  "Your function returned an object of {}".format(type(output)))

        self.assertTrue(len(output) == length, "Tuple does not have the required number of elements. "
                                               "Your function returned a {}-element tuple while the answer must "
                                               "be a {}-element tuple.".format(len(output), length))

    def check_float(self, value):
        is_float = isinstance(value, float)
        self.assertTrue(is_float, "Output is not a float. "
                                  "Your function returned an object of {}".format(type(value)))

    def check_int(self, value):
        is_int = isinstance(value, float)
        self.assertTrue(is_int, "Output is not a float. "
                                "Your function returned an object of {}".format(type(value)))

    def check_list(self, output, length=None):
        is_list = isinstance(output, list)
        self.assertTrue(is_list, "Output is not a list. "
                                 "Your function returned an object of {}".format(type(output)))

        if length is not None:
            self.assertTrue(length == len(output), "List does not have the required number of elements. "
                                                   "Your function returned a {}-element list while the answer must "
                                                   "be a {}-element list".format(len(output), length))

    def check_list_or_tuple(self, output, length=None):
        is_list_or_tuple = isinstance(output, list) or isinstance(output, tuple)
        self.assertTrue(is_list_or_tuple, "Output is not a list or tuple. "
                                          "Your function returned an object of {}".format(type(output)))

        if length is not None:
            self.assertTrue(length == len(output), "List or tuple does not have the required number of elements. "
                                                   "Your function returned a {}-element list or tuple while the "
                                                   "answer must be a {}-element list or tuple"
                            .format(len(output), length))

    def check_equal_values(self, output, reference, message, halt_on_equal):

        is_equal = np.allclose(output, reference)

        if halt_on_equal:
            is_equal = not is_equal

        self.assertTrue(is_equal, message)
