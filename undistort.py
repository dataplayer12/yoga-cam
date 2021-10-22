import cv2
import numpy as np
import time
import pickle

with open('dist_pickle.p','rb') as f:
	dist_pickle=pickle.load(f)

#print(dist_pickle.keys())
cmat=dist_pickle['mtx']
dcoeffs=dist_pickle['dist']
originalsize=(1616,1076)
newsize=(1920,1080)

newmat, roi=cv2.getOptimalNewCameraMatrix(cmat, dcoeffs, originalsize, 1, newsize)

newmat=np.array(newmat)
x,y,w,h=roi

src=cv2.VideoCapture(1)

ret,frame=src.read()
if ret:
	print(frame.shape)

#dst=cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))

fps=0

while ret:
	start=time.time()
	udist=cv2.undistort(frame, cmat, dcoeffs, None, newmat)
	#udist=cv2.fisheye.undistortImage(frame, cmat, dcoeffs, None, newmat, (1920,1080))
	udist = udist[y:y+h, x:x+w]
	udframe=cv2.resize(udist, (1280, 720))
	#dst.write(udframe)
	cv2.imshow('undistorted', udframe)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break
	ret,frame=src.read()
	end=time.time()
	fps=0.9*fps+0.1/(end -  start)
	#print(fps)

cv2.destroyAllWindows()
src.release()
#dst.release()