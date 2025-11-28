import numpy as np
import cv2 as cv
#create a black image
img = np.zeros((512, 512,3), dtype=np.uint8)
#Line drawn using cv.line() function
cv.line(img, (0, 511), (511, 0), (255, 0, 0), 5)
cv.line(img, (0, 0), (511, 511), (255, 255, 0), 5)
#Rectangle drawn using cv.rectangle() function
cv.rectangle(img, (212, 50), (312, 100), (0, 255, 0), 3)
#Circle drawn using cv.circle() function
cv.circle(img, (212, 212), 5, (0, 0, 225), -1)
#Display the image
cv.ellipse(img,(256,170),(50,50),0,270,180,255,-1)
#Text drawn using cv.putText() function
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'Teste',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
cv.imshow('Line', img)
cv.waitKey(0)