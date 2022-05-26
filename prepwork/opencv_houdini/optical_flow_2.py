#importing the required modules
import numpy as np
import cv2
#reading the input video whose optical flow is to be determined
videoread = cv2.VideoCapture("vid.mp4")
#defining the parameters to pass to the goodFeaturesToTrack() function as a dictionary
feature_points = dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)

#defining the parameters to pass to the calcOpticalFlowPyrLK() function as a dictionary
parameters = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#generating random colors to apply to the resulting 2D vectors
resultingvectorcolor = np.random.randint(0, 255, (100, 3))

#finding the feature points in the first frame using goodFeaturesToTrack() function
first, first_frame = videoread.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
points = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_points)

#mask image is created for drawing the vectors
resmask = np.zeros_like(first_frame)
while (1):
    second, second_frame = videoread.read()
    second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

    #optical flow is calculated using calcOpticalFlowPyrLK() function
    respoints, status, errors = cv2.calcOpticalFlowPyrLK(first_gray, second_gray,points, None,**parameters)

    #choosing the good points
    good_second = respoints[status == 1] 
    good_first = points[status == 1] #drawing the tracks

    for i, (first, second) in enumerate(zip(good_second,good_first)):
        A, B = second.ravel()
        C, D = first.ravel()
        resmask = cv2.line(resmask, (A, B), (C, D), resultingvectorcolor[i].tolist(), 2)

        second_frame = cv2.circle(second_frame, (A, B), 5, resultingvectorcolor[i].tolist(), -1)
        resvideo = cv2.add(second_frame, resmask)

        cv2.imshow('Result', resvideo)
        k = cv2.waitKey(25)
        if k == 27:
            break   

    #the previous frame and points are updated
    first_gray = second_gray.copy()
    points = good_second.reshape(-1, 1, 2)
    cv2.destroyAllWindows()
    videoread.release()