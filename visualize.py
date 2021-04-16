import cv2
import numpy as np
from func import *
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def VisualizeRotatingCaliper(polygon: np.ndarray, contour: np.ndarray):
    '''visualization ver.

    param:
        polygon:(n,n),the binary image of a polygon
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

    plt.ion()
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


        plt.imshow(polygon, cmap='gray')
        triangles = Triangulation([p1.x, p2.x, points[j].x], [p1.y, p2.y, points[j].y])
        plt.triplot(triangles)
        plt.pause(0.5)
        plt.clf()

        plt.show()

    # calculates the maximum length and 2 points on the convexhull
    p1, p2, p0 = max_dist_points
    max_res = [p1, p0] if euc_dist(p1, p0) > euc_dist(p2, p0) else [p2, p0]
    # calculates the minimum width and 2 points on the convexhull
    p1, p2, p0 = min_dist_points
    min_res = [p1, p0] if euc_dist(p1, p0) < euc_dist(p2, p0) else [p2, p0]
    return max_dist, min_dist, max_res, min_res


if __name__ == '__main__':
    image = cv2.imread('test.png', 0)
    contour, _ = cv2.findContours(image, 0, 1)
    contour = contour[0].reshape(-1, 2)

    max_dist, min_dist, max_res, min_res = VisualizeRotatingCaliper(image,contour)
