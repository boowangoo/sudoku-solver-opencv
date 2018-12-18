import cv2
import numpy as np

#load image
img = cv2.imread('homography-test.jpg', cv2.IMREAD_COLOR)

#corners of book covers (before)
frontCoverPtsBefore = np.array([[32, 48], [279, 136], [247, 430], [39, 281]], dtype="float32")
backCoverPtsBefore = np.array([[279, 136], [474, 36], [463, 316], [247, 430]], dtype="float32")

#corners of book covers (after)
frontCoverPtsAfter = np.array([[0, 0], [299, 0], [299, 599], [0, 599]], dtype="float32")
backCoverPtsAfter = np.array([[300, 0], [599, 0], [599, 599], [300, 599]], dtype="float32")

#get the transformation matrices for both covers
M_front = cv2.getPerspectiveTransform(frontCoverPtsBefore, frontCoverPtsAfter)
M_back = cv2.getPerspectiveTransform(backCoverPtsBefore, backCoverPtsAfter)

#warpPerspective both images
img_front = cv2.warpPerspective(img, M_front, (600, 600))
img_back = cv2.warpPerspective(img, M_back, (600, 600))

#copy half of the warped back cover into the warped front cover
np.copyto(img_front[:, 300:, :], img_back[:, 300:, :])

#display before and after
cv2.imshow('img', img)
cv2.imshow('img_front', img_front)
cv2.waitKey(0)
cv2.destroyAllWindows()