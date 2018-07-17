import cv2

frame = cv2.imread('400046.jpg')
image1 = cv2.GaussianBlur(frame,(0,0),3)
image2 = cv2.GaussianBlur(frame,(0,0),5)
image3 = cv2.GaussianBlur(frame,(0,0),3)
image4 = cv2.GaussianBlur(frame,(0,0),5)
cv2.addWeighted(frame,1.5,image1,-0.5,0,image1)
cv2.imshow('frame',frame)
cv2.imshow('image1',image1)

cv2.addWeighted(frame,1.5,image2,-0.5,0,image2)
#cv2.imshow('frame',frame)
cv2.imshow('image2',image2)

cv2.addWeighted(frame,2,image3,-1,0,image3)
#cv2.imshow('frame',frame)
cv2.imshow('image3',image3)

cv2.addWeighted(frame,2,image4,-1,0,image4)
#cv2.imshow('frame',frame)
cv2.imshow('image4',image4)

cv2.waitKey(0)
cv2.destroyAllWindows()