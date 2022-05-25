import cv2
import numpy as np
from matplotlib import pyplot as plt
# Our sketch generating function


def sketch(image):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Clean up image using Guassian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    
    # Extract edges
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)
    
    # Reverse white and black pixels
    # Do an invert binarize the image 
    #ret, thresh = cv2.threshold(canny_edges, 127, 255, cv2.THRESH_BINARY_INV)
    #ret, thresh = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    ret, thresh = cv2.threshold(canny_edges, 176, 255, 0)
    return canny_edges, ret, thresh
# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
path= "C:/Users/joris/OneDrive/Documenten/creatievemakers/summerschool/prepwork/opencv_houdini/OUT/footage_convex_hull_"
cap = cv2.VideoCapture(path+"%04d.jpg", cv2.CAP_IMAGES)

while True:
    ret, frame = cap.read()
    
    canny_edges, ret, thresh = sketch(frame)
    
    # Find contours 
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Sort Contors by area and then remove the largest frame contour
    n = len(contours) - 1
    contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]
    
    # Iterate through contours and draw the convex hull
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(frame, [hull], 0, (255, 0, 0), -1)
        cv2.drawContours(frame, [hull], 0, (255, 255, 0), 1)
        cv2.imshow('Our Live Sketcher', frame)

    #cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()   
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()  