
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: notebook.ipynb

import cv2 #Only to be used for Canny Edge Detector
import numpy as np
import test_simple as tests
Checker = tests.PS02Test()

def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.
    Args:
        img_in (numpy.array BGR): image containing a traffic light.
    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
        Numpy array: Height x Width matrix of Hough accumulator array  (Height and width from the image)
    """
    lowergray = np.array([10, 10, 10])
    uppergray = np.array([60, 60, 60])

    temp_img = np.copy(img_in)
    gray = cv2.inRange(temp_img, lowergray, uppergray)
    edges = cv2.Canny(img_in, 100, 50).astype(np.uint8)

    lines = cv2.HoughLinesP(edges, rho=1, theta=2*np.pi/180, threshold=30, minLineLength=20, maxLineGap=1)

#     matrix = generate_hough_accumulator(gray, edges)
    matrix = lines

    error = 5
    L0 = []
    A0 = []
    L60 = []
    A60 = []
    Ln60 = []
    An60 = []
    coordinates = (0, 0)

    if lines is not None:
        for line in lines:
            line = Line(line.flatten())

            if line.length < 500 and line.angle > (0 - error) and line.angle < (0 + error):
                A0.append(line.length)
                L0.append(line)

            if line.length < 500 and line.angle > (60 - error) and line.angle < (60 + error):
                A60.append(line.length)
                L60.append(line)


            if line.length < 500 and line.angle > (-60 - error) and line.angle < (-60 + error):
                An60.append(line.length)
                Ln60.append(line)

        if (len(L60) != 0):
            line60 = L60[np.argsort(A60)[-1]].line  ## 1
            line0 = L0[np.argsort(A0)[-1]].line
            linen60 = Ln60[np.argsort(An60)[-1]].line  ## 3

            mid60 = L60[np.argsort(A60)[-1]].mid
            mid0 = L0[np.argsort(A0)[-1]].mid
            midn60 = Ln60[np.argsort(An60)[-1]].mid

            x = int(mid0[0])
            upx = mid0[0]
            upy = mid0[1]
            botx = (line60[2] + linen60[0])/2
            boty = (line60[3] + linen60[1])/2
            y = int((boty - upy)/3 + upy)

            coordinates = (x, y)
            pixels = img_in[y, x, :]
            if pixels[0] > 220 and pixels[1] > 220 and pixels[2] > 220 :
                return (coordinates, matrix)
            else:
                return ((0,0), matrix)

    return (coordinates, matrix)

def StopSign(img_in,line):
    r, c, channel = img_in.shape
    error = 10
    upperx = int(line.mid[0] + error)
    uppery = int(line.mid[1] + error)
    lowerx = int(line.mid[0] - error)
    lowery = int(line.mid[1] - error)
    if ((img_in[uppery, upperx, 0] <15 and img_in[uppery, upperx, 1]<15)
     or (img_in[lowery, lowerx, 0]<15 and img_in[uppery, lowery, 1]<15)):
        return True
    else:
        return False
    return True

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.
    Args:
        img_in (numpy.array BGR): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
        Numpy array: Height x Width matrix of Hough accumulator array  (Height and width from the image)
    """



    lowergray = np.array([10, 10, 10])
    uppergray = np.array([60, 60, 60])

    temp_img = np.copy(img_in)
    gray = cv2.inRange(temp_img, lowergray, uppergray)
    edges = cv2.Canny(img_in, 100, 50).astype(np.uint8)

    lines = cv2.HoughLinesP(edges, rho=1, theta=2*np.pi/180, threshold=30, minLineLength=20, maxLineGap=1)

#     matrix = generate_hough_accumulator(gray, edges)
    matrix = lines
#     edges = cv2.Canny(img_in, 50, 100)
#     lines = cv2.HoughLinesP(edges, rho=1, theta=2*np.pi/180, threshold=30, minLineLength=20, maxLineGap=1)
#     matrix = generate_hough_accumulator(edges)

    Line_list = []
    A45 = []
    An45 = []
    error = 5
    coordinates = (0, 0)

    if lines is not None:
        for line in lines:
            line =  Line(line.flatten())


            if line.angle != 0 and StopSign(img_in,line) is True:
                Line_list.append(line)
                A45.append(np.abs(line.angle - 45))
                An45.append(np.abs(line.angle + 45))

        if (len(A45) < 2) or len(An45) < 2:
            return (coordinates, matrix)

        index = np.argsort(An45)
        line1 = Line_list[index[0]]
        line2 = Line_list[index[1]]
        if line1.angle < (-45 - error) or line1.angle > (-45 + error)  or line2.angle < (-45 - error) or line2.angle > (-45 + error) :
            return (coordinates, matrix)

        col = int((line1.mid[0] + line2.mid[0])/2)
        row = int((line1.mid[1] + line2.mid[1])/2)

        coordinates = (col, row)

    return (coordinates, matrix)


def ConsSign(img_in,line):
    r ,c , channel = img_in.shape
    error = 5
    upperx = int(line.mid[0] + error)
    uppery = int(line.mid[1] + error)
    lowerx = int(line.mid[0] - error)
    lowery = int(line.mid[1] - error)

    if ((img_in[uppery, upperx, 0] < 15 and img_in[uppery, upperx, 1] > 110 and img_in[uppery, upperx, 1] < 150 and img_in[uppery, upperx, 2] > 240)
     or (img_in[lowery, lowerx, 0] < 15 and img_in[uppery, lowery, 1] > 110 and img_in[uppery, upperx, 1] < 150 and img_in[uppery, upperx, 2] > 240 )):
        return True
    else:
        return False
    return True


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array BGR): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
        Numpy array: Height x Width matrix of Hough accumulator array  (Height and width from the image)
    """

    edges = cv2.Canny(img_in, 100, 50)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=30, maxLineGap=2)
#     matrix = generate_hough_accumulator(edges, edges)
    matrix = lines

    Line_list = []
    A45 = []
    An45 = []
    error = 5
    coordinates = (0, 0)

    if lines is not None:
        for line in lines:
            line =  Line(line.flatten())

            if line.length < 500 and line.angle != 0 and ConsSign(img_in,line) is True:
                Line_list.append(line)
                A45.append(np.abs(line.angle - 45))
                An45.append(np.abs(line.angle + 45))

        if (len(A45) == 0) or (len(An45) == 0):
            return (coordinates, matrix)

        index45 = np.argsort(A45)
        line1, line2 = Line_list[index45[0]], Line_list[index45[1]]

        indexn45 = np.argsort(An45)
        line3, line4 = Line_list[indexn45[0]], Line_list[indexn45[1]]

        col = (int((line1.mid[0] + line2.mid[0])/2) + int((line3.mid[0] + line4.mid[0])/2))//2 + 1
        row = (int((line1.mid[1] + line2.mid[1])/2) + int((line3.mid[1] + line4.mid[1])/2))//2 + 1

        coordinates = (col, row)

    return (coordinates, matrix)



def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.
    Args:
        img_in (numpy.array BGR): image containing a traffic light.
    Returns:
        (x,y) typle of the coordinates of the center of the sign.
        Numpy array: Height x Width matrix of Hough accumulator array  (Height and width from the image)
    """

    lowergray = np.array([10, 10, 10])
    uppergray = np.array([60, 60, 60])

    temp_img = np.copy(img_in)
    gray = cv2.inRange(temp_img, lowergray, uppergray)
    edges = cv2.Canny(img_in, 100, 50).astype(np.uint8)

    lines = cv2.HoughLinesP(edges, rho=1, theta=2*np.pi/180, threshold=30, minLineLength=20, maxLineGap=1)

#     matrix = generate_hough_accumulator(gray, edges)
    matrix = lines

    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30)
    coordinates = (0,0)
    res = (coordinates,matrix)

    if circles is not None:
        for circle in circles[0, :]:
            col = circle[0]
            row = circle[1]
            coordinates = (col, row)
            check = img_in[int(row), int(col), :]
            if check[0] == 255 and check[1] == 255 and check[2] == 255 :
                print(coordinates)
                res = (coordinates,matrix)
    return res

def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.
    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction
    Use these names for your output.
    See the instructions document for a visual definition of each
    sign.
    (Hint: Use all the functions defined above)
    Args:
        img_in (numpy.array BGR): input image containing at least one
                              traffic sign.
    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.
              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    dict = {}

    img_traffic = np.copy(img_in)
    img_notenter = np.copy(img_in)
    img_stop = np.copy(img_in)
    img_warn = np.copy(img_in)
    img_yields = np.copy(img_in)
    img_cons = np.copy(img_in)

    radii_range = range(10, 30, 1)
    (traffic,color),mat = traffic_light_detection(img_traffic,radii_range)
    notenter, mat = do_not_enter_sign_detection(img_notenter)
    stop, mat = stop_sign_detection(img_stop)
    warn, mat = warning_sign_detection(img_warn)
    yields, mat = yield_sign_detection(img_yields)
    cons, mat = construction_sign_detection(img_cons)



    if traffic is not None:
        dict['traffic_light'] = (traffic[0], traffic[1])
    if notenter != (0, 0):
        dict['no_entry'] = (notenter[0], notenter[1])
    if stop != (0,0):
        dict['stop'] = (stop[0], stop[1])
    if warn != (0,0):
        dict['warning'] = (warn[0], warn[1])
    if yields != (0,0):
        dict['yield'] = (yields[0], yields[1])
    if cons != (0,0):
        dict['construction'] = (cons[0], cons[1])


    print (dict)
    return dict

    raise NotImplementedError

def detectCircles(im, radius, useGradient = False):
    """
    Args:
        im (numpy.array RGB):the input image
        radius : specifies the radius of the circle
        useGradient: a flag that allows the user to optionally exploit the gradient direction measured at the edgepoints.
    (Caution: Your x,y maybe swapped)
    Returns:
        Numpy array: N x 2 matrix in which each row lists the (x,y) position of a detectedcircles’ center
        Numpy array: Height x Width matrix of Hough accumulator array  (Height and width from the image)

    """
    lowergray = np.array([10, 10, 10])
    uppergray = np.array([60, 60, 60])

    temp_img = np.copy(im)
    gray = cv2.inRange(temp_img, lowergray, uppergray)
    edges = cv2.Canny(gray, 25, 80)

    dx, dy = np.gradient(gray)
    gradientThetas = np.arctan2(-dy, dx)

    accu = np.zeros_like(gray)

    for index, currEdge in np.ndenumerate(edges):
        if currEdge != 0:
            r = radius
            x = index[1]
            y = index[0]
            for t in range(0,360,5):
                theta = np.radians(t)
                a = int(int(x - (r * np.cos(theta))))
                b = int(int(y + (r * np.sin(theta))))
                if (b in range(edges.shape[0])) and (a in range(edges.shape[1])):
                    accu[b,a] += 1

    maxaccum = np.max(accu)
    centers = np.transpose(np.nonzero(accu >= (maxaccum * .8)))
    center = np.zeros(centers.shape)
    center[:,0] = centers[:,1]
    center[:,1] = centers[:,0]
    aa = center[0:9,:]
    res = (aa, accu)

    return res

def  detectMultipleCircles(im, radius_min,radius_max):
    """
    Args:
        im (numpy.array RGB):the input image
        radius_min : specifies the minimum radius of the circle
        radius_max : specifies the maximum radius of the circle

    (Caution: Your x,y maybe swapped)
    Returns:
        Numpy array: N x 2 matrix in which each row lists the (x,y) position of a detectedcircles’ center
        Numpy array: Height x Width matrix of Hough accumulator array (Height and width from the image)

    """
#     lowergray = np.array([10, 10, 10])
#     uppergray = np.array([60, 60, 60])

#     temp_img = np.copy(im)
#     gray = cv2.inRange(temp_img, lowergray, uppergray)
#     edges = cv2.Canny(gray, 25, 80)

#     dx, dy = np.gradient(gray)
#     gradientThetas = np.arctan2(-dy, dx)

#     accu = np.zeros_like(gray)

#     for index, currEdge in np.ndenumerate(edges):
#         if currEdge != 0:
#             for r in range(radius_min, radius_max, 5):
#                 x = index[1]
#                 y = index[0]
#                 for t in range(0,360,5):
#                     theta = np.radians(t)
#                     a = int(int(x - (r * np.cos(theta))))
#                     b = int(int(y + (r * np.sin(theta))))
#                     if (b in range(edges.shape[0])) and (a in range(edges.shape[1])):
#                         accu[b,a] += 1

#     maxaccum = np.max(accu)
#     centers = np.transpose(np.nonzero(accu >= (maxaccum * .98)))
#     center = np.zeros(centers.shape)
#     center[:,0] = centers[:,1]
#     center[:,1] = centers[:,0]

#     res = (center, accu)

#     return res