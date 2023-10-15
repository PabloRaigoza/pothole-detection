import cv2
import numpy as np
 
# Read the original image
img = cv2.imread('FilterImages/8.jpg') 
img = cv2.resize(img, (640, 360))

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (15,15), 0)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=11) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection


def canny_filter(t1, t2):
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=t1, threshold2=t2) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

def canny_filter2(path, t1, t2):
    # Canny Edge Detection
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 360))
    grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayed_image, (15,15), 0)
    edges = cv2.Canny(image=blurred, threshold1=t1, threshold2=t2) # Canny Edge Detection
    winname = path
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



img_blur2 = cv2.GaussianBlur(img, (15,15), 0) 

hsv = cv2.cvtColor(img_blur2, cv2.COLOR_BGR2HSV)

lower_gray = np.array([0,0,0])
upper_gray = np.array([100,100,100])

mask = cv2.inRange(hsv , lower_gray, upper_gray)

for i in range(17, 50):
    path = "FilterImages/"+str(i)+".jpg"
    print("Showing: "+ path)
    canny_filter2(path, 20, 200)

cv2.destroyAllWindows()