#Drawing OpenCV Logo using OpenCV
import numpy as np
import cv2 as cv
#create a black image
img = np.zeros((512, 512,3), dtype=np.uint8)

#Drawing the red circle in OpenCV logo
cv.ellipse(img,(220,156),(50,50),0,115,420,( 0,0,255),-1)
cv.circle(img, (220,156), 25, (0,0,0), -1)   

#Drawing a green ellipse  to represent the green circle in OpenCV logo
cv.ellipse(img,(170,256),(50,50),0,0,300,(0,255,0),-1)
cv.circle(img, (170, 256), 25, (0,0,0), -1)
#Drawing  the blue circle in OpenCV logo
cv.ellipse(img,(290,256),(50,50),0,240,-60,(255,0,0),-1)
cv.circle(img, (290,256), 25, (0,0,0), -1)


#show the image
cv.imshow('OpenCV Logo', img)
cv.waitKey(0)
cv.destroyAllWindows()