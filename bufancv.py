import numpy as np
import math

def rhoThetaIntersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    #solve Ax = b
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([
        [rho1],
        [rho2]
    ])
    x, y = np.linalg.solve(A, b)
    x, y = int(np.round(x)), int(np.round(y))
    return (x, y)

# def groupClusterPts(ptList, thresh_dist):
#     for x1, y2 in vecList:
#         for x2, y2 in vecList:
#             if (x1 != x2 or y1 != y2) and math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < thresh_dist:

def xyIntersection(pt1, pt2, pt3, pt4):
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3
    x4, y4 = pt4

    denominatior = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    xNumerator = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    yNumerator = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    return (xNumerator // denominatior, yNumerator // denominatior)
