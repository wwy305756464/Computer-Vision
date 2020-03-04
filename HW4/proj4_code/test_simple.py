# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:37:39 2020

@author: venkata1996
"""

import numpy as np
import platform
import time
from verify_student_output import OutputCheck
from scene_generation import scene, stop_sign, traffic_light
from scene_generation import warning_sign, yield_sign, do_not_enter_sign
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import itertools


im = plt.imread('data/jupiter.jpg')
if platform.system() == 'Windows':
    NIX = False
    print("Running on Windows system")
else:
    NIX = True
    print("Running on Linux/OS X system")

def print_success_message(test_case):
    print("UnitTest {0} passed successfully!".format(test_case))
    
    
class PS02Test(OutputCheck):

    def testHoughtransform(self,detectCircles):
        """Test for Hough transform """
        #Dummy centers included 
        centers = np.array([[74,235]])#, [219,320], [106,458], [289,213],  [455,587]])   MATCH RADIUS WITH CENTERS WHILE DEBUGGING
        radius_as_pixels = 14#,32,53,111,160
        use_gradient = False
        output,hough_space =  detectCircles(im,radius_as_pixels,use_gradient)
        
        row1,column1 = centers.shape
        row2,column2 = output.shape
        
        
        
        self.assertEqual(column2,2,msg="Return numpy matrix should be Nx2") 
        self.assertLessEqual(row2,10,msg="More than 10 centers returned") 
        checky = False
        check_out = np.empty((0,2))
        for i0, i1 in itertools.product(np.arange(centers.shape[0]),np.arange(output.shape[0])):
            if np.all(np.isclose(centers[i0], output[i1], atol=5)):
                check_out = np.concatenate((check_out, [output[i1]]), axis=0)
        if check_out.shape[0]>0:
            checky = True    
            fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True)
            ax1.set_aspect('equal')

            # Show the image
            ax1.imshow(im)
            ax2.imshow(hough_space)
            for q in range(row2):
               
                circ = Circle((output[q,0],output[q,1]),radius_as_pixels, color='red', fill = False, linewidth=2)
                ax1.add_patch(circ)
            plt.show()
        self.assertTrue(checky,msg="None of the centers returned matched the original ones") 
        print_success_message("testHoughstransform")
        
    def testHoughtransformwithgradient(self,detectCircles):
        """Test for Hough transform """
        #Dummy centers included 
        centers = np.array([[74,235]])#, [219,320], [106,458], [289,213],  [455,587]])  MATCH RADIUS WITH CENTERS WHILE DEBUGGING
        radius_as_pixels = 14#,32,53,111,160
        use_gradient = True
        start = time.time()
        output,hough_space =  detectCircles(im,radius_as_pixels,use_gradient)
        end = time.time()
        time_elapsed = end - start
        print("Time elapsed:",time_elapsed)
        self.assertLessEqual(time_elapsed,120,msg="Time elapsed must be less than 180 seconds")


        row1,column1 = centers.shape
        row2,column2 = output.shape
 
        self.assertEqual(column2,2,msg="Return numpy matrix should be Nx2") 

        checky = False
        check_out = np.empty((0,2))
        for i0, i1 in itertools.product(np.arange(centers.shape[0]),np.arange(output.shape[0])):
            if np.all(np.isclose(centers[i0], output[i1], atol=5)):
                check_out = np.concatenate((check_out, [output[i1]]), axis=0)
        if check_out.shape[0]>0:
            checky = True  
            fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True)
            ax1.set_aspect('equal')

            # Show the image
            ax1.imshow(im)
            ax2.imshow(hough_space)
            for q in range(row2):
                circ = Circle((output[q,0],output[q,1]),radius_as_pixels, color='red', fill = False, linewidth=2)
                ax1.add_patch(circ)
            plt.show()
        self.assertTrue(checky,msg="None of the centers returned matched the original ones") 
        print_success_message("testHoughstransform_withgradient")
        
    def testHoughtransformMultiple(self,detectMultipleCircles):
        """Test for Hough transform """
        # Dummy centers included 
        centers = np.array([[74,235], [219,320], [106,458], [289,213],  [455,587]])
        radius_as_pixels_max = 170
        radius_as_pixels_min = 10
        output,hough_space =  detectMultipleCircles(im,radius_as_pixels_min,radius_as_pixels_max)
        row1,column1 = centers.shape
        row2,column2 = output.shape
 
        self.assertEqual(column2,2,msg="Return numpy matrix should be Nx2") 
        checky = False
        check_out = np.empty((0,2))
        for i0, i1 in itertools.product(np.arange(centers.shape[0]),np.arange(output.shape[0])):
            if np.all(np.isclose(centers[i0], output[i1], atol=5)):
                check_out = np.concatenate((check_out, [output[i1]]), axis=0)
        if check_out.shape[0]>0:
            checky = True    
            fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True)
            ax1.set_aspect('equal')

            # Show the image
            ax1.imshow(im)
            ax2.imshow(hough_space)
            
            ax1.scatter(output[:,0],output[:,1], color='red')
            plt.show()
        self.assertTrue(checky,msg="None of the centers returned matched the original ones") 
        print("{} out of total 5 circles found".format(check_out.shape[0]))
        print_success_message("testHoughstransformMultiple")
        
        
    def traffic_light_scene_helper(self, scene_dims, t_sign_fn, t_sign_size,
                                   detect_fn, n_iter, scene_type, tol = 5,flag=1):
        radii_range = range(10, 30, 1)
        status = ["red", "green", "yellow"]
        for i in range(n_iter):  # Multiple iterations to prevent lucky pass
            if(scene_type == "blank"):
                test_canvas = scene.BlankScene(scene_dims)
            else:
                test_canvas = scene.Scene(scene_dims)

            expected_sign = t_sign_fn(t_sign_size, status[i % 3])

            top_left = (224,224)
            test_canvas.place_sign(top_left, expected_sign)

            all_info = test_canvas.get_objects()
            expected_x = all_info[expected_sign.name]['x']
            expected_y = all_info[expected_sign.name]['y']

            test_image = (255 * test_canvas.get_scene()).astype(np.uint8)
           
           
            if flag==1:
                result,hough_space = detect_fn(test_image, radii_range)
                (x,y),c = result
             
                
                fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True)
                ax1.set_aspect('equal')
    
                # Show the image
                ax1.imshow(test_image)
                ax2.imshow(hough_space)
                
                ax1.scatter(x,y,color = 'r')
                ax1.text(x,y,"Center {}".format(result))
                print(result)
                plt.show()
                self.check_nonetype(result)
                self.check_tuple(result, 2)
    
                self.assertAlmostEqual(expected_x, result[0][0], delta=tol,
                                       msg="X coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_x, result[0][0], tol))
    
                self.assertAlmostEqual(expected_y, result[0][1], delta=tol,
                                       msg="Y coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_y, result[0][1], tol))
    
                self.assertEqual(result[1], status[i % 3],
                                 msg="Wrong state value. Expected: {}. "
                                     "Returned: {}".format(status[i % 3],
                                                           result[1]))
            else:
                result1 = detect_fn(test_image)
                print(result1)
                result = result1['traffic_light']
                (x,y) = result
                self.check_nonetype(result)
                self.check_tuple(result, 2)
    
                self.assertAlmostEqual(expected_x, result[0], delta=tol,
                                       msg="X coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_x, result[0], tol))
    
                self.assertAlmostEqual(expected_y, result[1], delta=tol,
                                       msg="Y coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_y, result[1], tol))
    
  
                
        
     

    def traffic_sign_scene_helper(self, scene_dims, t_sign_fn, t_sign_size,
                                  detect_fn, n_iter, tol=5,flag = 1):

        for i in range(n_iter):  # Multiple iterations to prevent lucky pass
            test_canvas = scene.BlankScene(scene_dims)

            expected_sign = t_sign_fn(t_sign_size)

            top_left = (224,224)
            test_canvas.place_sign(top_left, expected_sign)

            all_info = test_canvas.get_objects()
            expected_x = all_info[expected_sign.name]['x']
            expected_y = all_info[expected_sign.name]['y']

            test_image = (255 * test_canvas.get_scene()).astype(np.uint8)
          
            if flag==1:
                result,hough_space = detect_fn(test_image)
                (x,y) = result
                
                fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True)
                ax1.set_aspect('equal')
    
                # Show the image
                ax1.imshow(test_image)
                ax2.imshow(hough_space)
                
                ax1.scatter(x,y,color = 'r')
                ax1.text(x,y,"Center {}".format(result))
                print(result)
                plt.show()
                self.check_nonetype(result)
                self.check_tuple(result, 2)
    
                self.assertAlmostEqual(expected_x, result[0], delta=tol,
                                       msg="X coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_x, result[0], tol))
    
                self.assertAlmostEqual(expected_y, result[1], delta=tol,
                                       msg="Y coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_y, result[1], tol))
            if flag==2 or flag==3:
                result1 = detect_fn(test_image)
                print(result1)
                if flag==2:
                    result = result1['yield']
                if flag==3:
                    result = result1['no_entry']
                (x,y) = result
                
                
                self.check_nonetype(result)
                self.check_tuple(result, 2)
    
                self.assertAlmostEqual(expected_x, result[0], delta=tol,
                                       msg="X coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_x, result[0], tol))
    
                self.assertAlmostEqual(expected_y, result[1], delta=tol,
                                       msg="Y coordinate does not meet tolerance. "
                                           "Expected: {}. Returned: {}. "
                                           "Tolerance: {}."
                                           "".format(expected_y, result[1], tol))
                

    #@weight(20)  # Comment this when debugging
    def testStopSignBlank(self,stop_sign_detection):
        """Test for stop sign detection with a blank scene (20 points)"""
        self.traffic_sign_scene_helper((500, 500), stop_sign.StopSign,
                                       100,   stop_sign_detection, 1, 10)
        print_success_message("testStopSignBlank")

    #@weight(5)  # Comment this when debugging
    def testWarningSignBlank(self,warning_sign_detection):
        """Test for warning sign detection with a blank scene"""

        self.traffic_sign_scene_helper((500, 500), warning_sign.WarningSign,
                                       100,   warning_sign_detection, 1)
        print_success_message("testWarningSignBlank")

    #@weight(5)  # Comment this when debugging
    def testConstructionSignBlank(self,construction_sign_detection):
        """Test for construction sign detection with a blank scene"""

        self.traffic_sign_scene_helper((500, 500), warning_sign.ConstuctionSign,
                                       100,   construction_sign_detection, 1)
        print_success_message("testConstructionSignBlank")

    #@weight(10)  # Comment this when debugging
    def testDoNotEnterSignBlank(self,do_not_enter_sign_detection):
        """Test for do not enter sign detection with a blank scene"""

        self.traffic_sign_scene_helper((500, 500), do_not_enter_sign.DoNotEnterSign,
                                       100,   do_not_enter_sign_detection, 1)
        print_success_message("testDoNotEnterSignBlank")

    #@weight(20)
    def testYieldSignBlank(self,yield_sign_detection):
        """Test for yield sign detection with a blank scene"""
        self.traffic_sign_scene_helper((500, 500), yield_sign.YieldSign,
                                       100,   yield_sign_detection, 1, 10)
        print_success_message("testYieldSignBlank")

    #@weight(15)
    def testTrafficLightBlank(self,traffic_light_detection):
        """Test for traffic light detection with a blank scene"""
        self.traffic_light_scene_helper((500, 500), traffic_light.TrafficLight,
                                        30,  traffic_light_detection, 1,
                                        "blank")
        print_success_message("testTrafficLightBlank")
      
    #@weight(25)
    def testTrafficLightScene(self,traffic_light_detection):
        """Test for traffic light detection with a simulated street scene"""
        self.traffic_light_scene_helper((500, 500), traffic_light.TrafficLight,
                                        30,   traffic_light_detection, 1,
                                        "scene")
        print_success_message("testTrafficLightScene")
        
    def testTrafficSignScene(self,traffic_sign_detection):
        """Test for multiple traffic signs and lights detection with a simulated street scene"""
        self.traffic_light_scene_helper((500, 500), traffic_light.TrafficLight,
                                        30,   traffic_sign_detection, 1,
                                        "scene",5,0)
        # self.traffic_sign_scene_helper((500, 500), yield_sign.YieldSign,
        #                                100,  traffic_sign_detection, 1, 10,2)
        self.traffic_sign_scene_helper((500, 500), do_not_enter_sign.DoNotEnterSign,
                                       100,   traffic_sign_detection, 1,5,3)
        print_success_message("testTrafficSignScene")
    
