# HOW TO USE
# Use this with a printed SUDOKU Grid
# Press ESC key to Exit
import cv2
import numpy as np

ratio2 = 3
kernel_size = 3
lowThreshold = 30

frame = cv2.resize(cv2.imread('sudoku-test.jpg', cv2.IMREAD_COLOR), (600, 600))

 # Preprocess image, convert from RGB to Gray
sudoku1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
sudoku1 = cv2.blur(sudoku1, (3,3))
# Apply Canny edge detection
edges = cv2.Canny(sudoku1, lowThreshold, lowThreshold*ratio2, kernel_size)
# Apply Hough Line Transform, return a list of rho and theta
lines = cv2.HoughLines(edges, 2, np.pi/180, 300, 0, 0)
if (lines is not None):
    lines = lines[0]
    lines = sorted(lines, key=lambda line:line[0])
    # Define the position of horizontal and vertical line
    pos_hori = 0
    pos_vert = 0
    for rho,theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        # If b > 0.5, the angle must be greater than 45 degree
        # so we consider that line as a vertical line
        if (b>0.5):
        # Check the position
            if(rho-pos_hori>10):
            # Update the position
                pos_hori=rho
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        else:
            if(rho-pos_vert>10):
                pos_vert=rho
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

# Show the result        
cv2.imshow("SUDOKU Solver", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()