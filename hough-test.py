import cv2
import numpy as np
import math

img = cv2.imread('sudoku-test3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresh_type = cv2.THRESH_BINARY_INV
bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)

# find puzzle contour and corner points
conts = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

maxContSize = 0
for cont in conts:
    # assuming puzzle contour is the one with the largest area
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

# find hough lines, and their points of intersection
rho, theta, thresh = 2, np.pi/180, 400
lines = cv2.HoughLines(bin_img, rho, theta, thresh)
hlines = []
vlines = []
for rho,theta in lines[:, 0, :]:
    # print("theta: " + str(theta))
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    if abs(theta - math.radians(0)) < math.radians(5) or abs(theta - math.radians(180)) < math.radians(5):
        vlines.append([rho, theta])
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    elif abs(theta - math.radians(90)) < math.radians(5) or abs(theta - math.radians(270)) < math.radians(5):
        hlines.append([rho, theta])
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)

cv2.drawContours(img, [maxCont], 0, (0, 0, 0), 3)

for hline in hlines:
    for vline in vlines:
        intersection = isIntersecting(hline, vline)
        if intersection != None:
            



cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()