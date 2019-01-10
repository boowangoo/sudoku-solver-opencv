import cv2
import numpy as np
import math
import sys

import bufancv as bfcv

fNum = sys.argv[1]
filename = "puzzles/puzzle" + str(fNum) + ".jpg"
# filename = "puzzle2.jpg"

img = cv2.imread(filename, cv2.IMREAD_COLOR)
img_resized = cv2.resize(img, (600, 600))
img_grayscale = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
img_denoised = cv2.fastNlMeansDenoising(img_grayscale, None, 9, 13)
img_thresh_gaussian = cv2.adaptiveThreshold(img_denoised, 233,\
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

keypoints = np.zeros((4, 4, 2))

conts = cv2.findContours(img_thresh_gaussian, cv2.RETR_EXTERNAL,\
        cv2.CHAIN_APPROX_NONE)[1]

maxContSize = 0
for cont in conts:
    contSize = cv2.contourArea(cont)
    if contSize > maxContSize:
        maxContSize = contSize
        maxCont = cont

contSums = maxCont.sum(axis = (1, 2))
topL = maxCont[np.argmin(contSums), 0, :]
botR = maxCont[np.argmax(contSums), 0, :]

contDiffs = np.diff(maxCont[:, 0, :], axis = 1)
topR = maxCont[np.argmin(contDiffs), 0, :]
botL = maxCont[np.argmax(contDiffs), 0, :]

# convert maxCont into an image so it can be warped
img_cont = np.zeros((600, 600))
cv2.drawContours(img_cont, [maxCont], 0, (255, 255, 255), thickness=1)

oldRect = np.array([topL, topR, botR, botL], dtype = "float32")
newRect = np.array([[10, 10], [590, 10], [590, 590], [10, 590]], dtype = "float32")

M = cv2.getPerspectiveTransform(oldRect, newRect)
img_warped = cv2.warpPerspective(img_resized, M, (600, 600))
img_warped_thresh = cv2.warpPerspective(img_thresh_gaussian, M, (600, 600))
img_warped_cont = cv2.warpPerspective(img_cont, M, (600, 600))


img_edges = cv2.Canny(img_warped_thresh, 100, 200, apertureSize = 3)

lines = cv2.HoughLines(img_edges, 2, math.radians(1), 200)
vlines = []
hlines = []
for rho,theta in lines[:, 0, :]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 3000*(-b))
    y1 = int(y0 + 3000*(a))
    x2 = int(x0 - 3000*(-b))
    y2 = int(y0 - 3000*(a))

    if abs(theta - math.radians(0)) < math.radians(1) or\
            abs(theta - math.radians(180)) < math.radians(1):
        # vertical lines
        vlines.append((rho, theta))
    elif abs(theta - math.radians(90)) < math.radians(1):
        # horizontal lines
        hlines.append((rho, theta))

intPts = np.zeros((600, 600), dtype=bool)
for hline in hlines:
    for vline in vlines:
        intPt = bfcv.rhoThetaIntersection(hline, vline)
        if intPt[1] < 600 and intPt[0] < 600:
            intPts[intPt[1], intPt[0]] = True
            # img_warped[intPt[1], intPt[0], :] = (0, 255, 0)

def medianOfRect(arr2d, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = 0 if x1 < 0 else x1
    x2 = 599 if x2 > 599 else x2
    y1 = 0 if y1 < 0 else y1
    y2 = 599 if y2 > 599 else y2
    
    xList = []
    yList = []

    for xi, yi in np.ndindex(x2 - x1, y2 - y1):
        if arr2d[y1 + yi, x1 + xi]:
            xList.append(x1 + xi)
            yList.append(y1 + yi)

    xMedian = None if len(xList) == 0 else int(np.median(np.array(xList)))
    yMedian = None if len(yList) == 0 else int(np.median(np.array(yList)))
    
    if xMedian == None or yMedian == None:
        return None
    else:
        return (xMedian, yMedian)



keyPts = np.array([
    [(10 ,  10), (203,  10), (397,  10), (590,  10)],
    [(10 , 203), (203, 203), (397, 203), (590, 203)],
    [(10 , 397), (203, 397), (397, 397), (590, 397)],
    [(10 , 590), (203, 590), (397, 590), (590, 590)]
])
keyPtsFlg = np.zeros((4, 4), dtype=bool)
keyPtsFlg[0, 0], keyPtsFlg[3, 3], keyPtsFlg[0, 3], keyPtsFlg[3, 0] = True, True, True, True

for i in range(600):
    j = 600 - 1 - i

    if (keyPtsFlg[0, 1], keyPtsFlg[0, 2], keyPtsFlg[1, 0], keyPtsFlg[2, 0],
            keyPtsFlg[3, 1], keyPtsFlg[3, 2], keyPtsFlg[1, 3], keyPtsFlg[2, 3]) ==\
            (True, True, True, True, True, True, True, True):
        break

    if not keyPtsFlg[0, 1] and img_warped_cont[i, 203] > 0.0:
        keyPts[0, 1, 1] = i
        medPt = medianOfRect(intPts, keyPts[0, 1] - [20, 5], keyPts[0, 1] + [20, 20])
        if medPt != None:
            keyPts[0, 1] = medPt
        keyPtsFlg[0, 1] = True
        # cv2.circle(img_warped, medPt, 3, (0, 0, 255))
    if not keyPtsFlg[0, 2] and img_warped_cont[i, 397] > 0.0:
        keyPts[0, 2, 1] = i
        medPt = medianOfRect(intPts, keyPts[0, 2] - [20, 5], keyPts[0, 2] + [20, 20])
        if medPt != None:
            keyPts[0, 2] = medPt
        keyPtsFlg[0, 2] = True
        # cv2.circle(img_warped, medPt, 3, (0, 0, 255))

    if not keyPtsFlg[1, 0] and img_warped_cont[203, i] > 0.0:
        keyPts[1, 0, 0] = i
        medPt = medianOfRect(intPts, keyPts[1, 0] - [5, 20], keyPts[1, 0] + [20, 20])
        if medPt != None:
            keyPts[1, 0] = medPt
        keyPtsFlg[1, 0] = True
    if not keyPtsFlg[2, 0] and img_warped_cont[397, i] > 0.0:
        keyPts[2, 0, 0] = i
        medPt = medianOfRect(intPts, keyPts[2, 0] - [5, 20], keyPts[2, 0] + [20, 20])
        if medPt != None:
            keyPts[2, 0] = medPt
        keyPtsFlg[2, 0] = True

    if not keyPtsFlg[3, 1] and img_warped_cont[j, 203] > 0.0:
        keyPts[3, 1, 1] = j
        medPt = medianOfRect(intPts, keyPts[3, 1] - [20, 20], keyPts[3, 1] + [20, 5])
        if medPt != None:
            keyPts[3, 1] = medPt
        keyPtsFlg[3, 1] = True
    if not keyPtsFlg[3, 2] and img_warped_cont[j, 397] > 0.0:
        keyPts[3, 2, 1] = j
        medPt = medianOfRect(intPts, keyPts[3, 2] - [20, 20], keyPts[3, 2] + [20, 5])
        if medPt != None:
            keyPts[3, 2] = medPt
        keyPtsFlg[3, 2] = True

    if not keyPtsFlg[1, 3] and img_warped_cont[203, j] > 0.0:
        keyPts[1, 3, 0] = j
        medPt = medianOfRect(intPts, keyPts[1, 3] - [20, 20], keyPts[1, 3] + [5, 20])
        if medPt != None:
            keyPts[1, 3] = medPt
        keyPtsFlg[1, 3] = True
    if not keyPtsFlg[2, 3] and img_warped_cont[397, j] > 0.0:
        keyPts[2, 3, 0] = j
        medPt = medianOfRect(intPts, keyPts[2, 3] - [20, 20], keyPts[2, 3] + [5, 20])
        if medPt != None:
            keyPts[2, 3] = medPt
        keyPtsFlg[2, 3] = True

keyPts[1, 1] = bfcv.xyIntersection(keyPts[0, 1], keyPts[3, 1], keyPts[1, 0], keyPts[1, 3])
keyPts[1, 1] = medianOfRect(intPts, keyPts[1, 1] - [20, 20], keyPts[1, 1] + [20, 20])

keyPts[1, 2] = bfcv.xyIntersection(keyPts[0, 2], keyPts[3, 2], keyPts[1, 0], keyPts[1, 3])
keyPts[1, 2] = medianOfRect(intPts, keyPts[1, 2] - [20, 20], keyPts[1, 2] + [20, 20])

keyPts[2, 1] = bfcv.xyIntersection(keyPts[0, 1], keyPts[3, 1], keyPts[2, 0], keyPts[2, 3])
keyPts[2, 1] = medianOfRect(intPts, keyPts[2, 1] - [20, 20], keyPts[2, 1] + [20, 20])

keyPts[2, 2] = bfcv.xyIntersection(keyPts[0, 2], keyPts[3, 2], keyPts[2, 0], keyPts[2, 3])
keyPts[2, 2] = medianOfRect(intPts, keyPts[2, 2] - [20, 20], keyPts[2, 2] + [20, 20])

# print(keyPts)

# for x in range(4):
#     for y in range(4):
#         cv2.circle(img_warped, tuple(keyPts[x, y]), 3, (0, 0, 255))

# ======
cells = []
newRect = np.array([[0, 0], [299, 0], [299, 299], [0, 299]], dtype = "float32")

for i in range(3):
    for j in range(3):
        oldRect = np.array([keyPts[0+i, 0+j], keyPts[0+i, 1+j], keyPts[1+i, 1+j], keyPts[1+i, 0+j]], dtype = "float32")
        M = cv2.getPerspectiveTransform(oldRect, newRect)
        img_thirds = cv2.warpPerspective(img_warped_thresh, M, (300, 300))
        cells.append(img_thirds[0:99, 0:99])
        cells.append(img_thirds[100:199, 0:99])
        cells.append(img_thirds[200:299, 0:99])
        cells.append(img_thirds[0:99, 100:199])
        cells.append(img_thirds[100:199, 100:199])
        cells.append(img_thirds[200:299, 100:199])
        cells.append(img_thirds[0:99, 200:299])
        cells.append(img_thirds[100:199, 200:299])
        cells.append(img_thirds[200:299, 200:299])



for i in range(81):
    writeName = "dataset/cell_" + str(i) + "_" + str(fNum) + ".jpg"
    # print(writeName)
    cv2.imwrite(writeName, cells[i])
# cv2.imshow('cell', cells[0])


# cv2.imshow('img', cells[1])
# cv2.imshow('img', img_warped_cont)
# cv2.imshow('img1', img_warped)
# cv2.imshow('img2', img_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()