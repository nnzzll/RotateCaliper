import cv2
import numpy as np


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


def subtract(p1: Point, p2: Point) -> Point:
    return Point(p1.x-p2.x, p1.y-p2.y)


def add(p1: Point, p2: Point) -> Point:
    return Point(p1.x+p2.x, p1.y+p2.y)


def cross(p1: Point, p2: Point, p0: Point) -> float:
    '''calculate area of Parallelogram formed by the three points.'''
    vector1 = subtract(p1, p0)
    vector2 = subtract(p2, p0)
    return abs(vector1.x*vector2.y-vector1.y*vector2.x)


def euc_dist(p1: Point, p2: Point) -> float:
    return np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)


def createPointList(array: np.ndarray) -> list:
    '''Convert an array to List[Point]

    param:
        array:(N,2) ,xy shape
    '''
    PointList = []
    for xy in array:
        point = Point(xy[0], xy[1])
        PointList.append(point)
    return PointList


def findInitialPoints(array: np.ndarray) -> int:
    '''calculate the initial points for RotatingCaliper algorithm

    param:
        array:(N,2) ,xy shape
    '''

    x = array[:, 0]
    y = array[:, 1]
    idx_max = np.where(y == y.max())[0][0]
    idx_min = np.where(y == y.min())[0][0]
    return idx_max, idx_min


def RotatingCaliper(contour: np.ndarray):
    '''Based on RotatingCaliper Algorithm,
    calculates the maximum length and minimum width of a polygon.

    param:
        contour:(N,2),xy shape
    '''
    # Calculates the convexhull of the contour.
    convexhull = cv2.convexHull(contour).reshape(-1, 2)
    points = createPointList(convexhull)
    # Find the max/min index of y-axis as the initial points.
    idx_max, idx_min = findInitialPoints(convexhull)
    # Create a index list.
    idx_list = [i for i in range(idx_min, len(convexhull))] + \
        [i for i in range(0, idx_min)]
    # List[Point] to save points when their distance is maximum or minimum.
    max_dist_points = [Point(), Point(), Point()]
    min_dist_points = [Point(), Point(), Point()]
    # Initialization
    j = idx_max
    max_dist = 0
    min_dist = np.inf

    for i in idx_list:
        # p1-p2:2 point line,p0:moving point.
        p1 = points[i]
        p2 = points[i+1] if i+1 <= max(idx_list) else points[0]
        p0 = points[j]
        max_area = 0
        area = cross(p1, p2, p0)
        # find the max area,which means p1-p2 and a parallel line across p0
        # contains the whole polygon area
        while(area >= max_area):
            max_area = area
            # find the maximum distance between 2 points
            _max_dist = max(euc_dist(p1, p0), euc_dist(p2, p0))
            if _max_dist > max_dist:
                max_dist = _max_dist
                max_dist_points[0] = p1
                max_dist_points[1] = p2
                max_dist_points[2] = p0

            # move p0
            j = j+1 if j < max(idx_list) else 0
            p0 = points[j]
            area = cross(p1, p2, p0)

        j = j-1 if j > 0 else max(idx_list)

        # find the minimum distance between 2 points
        _min_dist = min(euc_dist(p1, points[j]), euc_dist(p2, points[j]))
        if _min_dist < min_dist:
            min_dist = _min_dist
            min_dist_points[0] = p1
            min_dist_points[1] = p2
            min_dist_points[2] = points[j]

    # calculates the maximum length and 2 points on the convexhull
    p1, p2, p0 = max_dist_points
    max_res = [p1, p0] if euc_dist(p1, p0) > euc_dist(p2, p0) else [p2, p0]
    # calculates the minimum width and 2 points on the convexhull
    p1, p2, p0 = min_dist_points
    min_res = [p1, p0] if euc_dist(p1, p0) < euc_dist(p2, p0) else [p2, p0]
    return max_dist, min_dist, max_res, min_res
