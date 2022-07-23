# get_contours.py
# study with getcontours() command
# 7/22/22

import cv2

img = cv2.imread('./images/people2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_thresh = img_gray.copy()
cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV, dst=img_thresh)

# cv2.dilate(img_thresh, (13,13), img_thresh, iterations=25)
# cv2.imshow('gray', img_gray)
# cv2.imshow('thresh', img_thresh)

contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (0,255,0), 4)

for contour in contours:
    # get area of contour
    area_threshold = 800
    # if area is too small, skip it, don't draw boxes around it
    if cv2.contourArea(contour) > area_threshold:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y), (x+w, y+h), (5,0,255), 2)

cv2.imshow('final output', img)
cv2.waitKey()



