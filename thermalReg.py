"""
Attempts to find and fix affine warping between thermal cameras and returns affine
matrix which describes the transform which will fix them.

Attempts to align input image 2 to image 1.

::Usage::
		aligned,warp_matrix = translationalThermalReg(img1,img2)

img1 and img2 should be grayscale and of the same size NxM.


Alex Fafard et. al.
Rochester Insitute of Technology
1/31/2017
Graduate Laboratory 

Contact Ajf4163@rit.edu with Questions

"""

def translationalThermalReg(im1,im2):
	import cv2,numpy

	#get dimensions
	s1=im1.shape
	s2=im2.shape

	#check sizes agree as a sanity check for inputs

	if s1!=s2:
		raise TypeError('Array Inputs are of different sizes!')

	#Select translation model in CV
	warp_model = cv2.MOTION_AFFINE

	#Define 2x3 Warp Matrix
	warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)

	#Number of iterations allowed to converge on solution
	num_it=10000

	#Terminal Threshold
	termTh = 1e-9

	#Define Stopping Criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_it,  termTh)

	#Ensure images are of datatype float32 (for compatibility with transformation convergence)
	im1=im1.astype(numpy.float32)
	im2=im2.astype(numpy.float32)

	#Find Ideal Transform given input parameters
	(cc, warp_matrix) = cv2.findTransformECC(im1,im2,warp_matrix, warp_model, criteria)
 
 	#Apply Transform
	aligned = cv2.warpAffine(im2, warp_matrix, (s1[1], s1[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
	print('Calculated Affine Warp Matrix:')
	print(warp_matrix)

	return aligned, warp_matrix
 

#Test Harness for debugging and testing of functions 
if __name__=="__main__":
	import PIL.Image, requests, cv2, numpy, StringIO

	#Some Magic, to pull down an image from the web
	url ='https://cdn.pixabay.com/photo/2015/08/14/19/42/frog-888798_1280.jpg'
	response = requests.get(url)
	img = numpy.array(PIL.Image.open(StringIO.StringIO(response.content)))
	
	#Ensure Grayscale
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img=img.astype(numpy.float)

	arr = numpy.zeros([img.shape[0]+500,img.shape[1]+500])	

#Create a copy of the image and a translated & warped version.
	img1 = arr.copy()
	img1[:-500,:-500]=img


	rows,cols = arr.shape
	
	#Store Points to Deform the image
	pts1 = numpy.float32([[300,300],[200,50],[500,600]])
	pts2 = numpy.float32([[300,288],[190,50],[480,570]])
	M = cv2.getAffineTransform(pts1,pts2)
	
	img2 = arr.copy()
	img2[250:-250,250:-250]=img

	#Apply Affine Transform
	img2= cv2.warpAffine(img2,M,(cols,rows))

	#Try to fix it.
	aligned,warp_matrix = translationalThermalReg(img1,img2)

	Inputmosaic = numpy.uint8(numpy.hstack([img1,img2]))
	Outputmosaic = numpy.uint8(numpy.hstack([img1,aligned]))

	#Display Results
	cv2.namedWindow('Inputs',cv2.WINDOW_NORMAL)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(Inputmosaic,'Image 1',(700,1200), font, 5,(255,255,255),2)
	cv2.putText(Inputmosaic,'Image 2',(2300,1200), font, 5,(255,255,255),2)
	cv2.imshow('Inputs',Inputmosaic)

	cv2.namedWindow('Outputs',cv2.WINDOW_NORMAL)
	cv2.putText(Outputmosaic,'Image 1',(700,1200), font, 5,(255,255,255),2)
	cv2.putText(Outputmosaic,'Image 2',(2300,1200), font, 5,(255,255,255),2)
	cv2.imshow('Outputs',Outputmosaic)
#	cv2.imwrite('Output.png',Outputmosaic)
#	cv2.imwrite('Input.png',Inputmosaic)

	cv2.waitKey(0)


