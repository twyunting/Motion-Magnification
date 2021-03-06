# -*- coding: utf-8 -*-
"""Homework4_YuntingChiu.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nehVMUtJ1-F6fD8Jrb6b5ir_unBC91MS

# Homework
Question 1 (20pts): Basic Video Processing. You can see the tutorials on video processing:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
"""

#from google.colab import files
#upload = files.upload()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy import angle, real
from numpy import exp, abs, pi, sqrt
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
import imageio

cap = cv2.VideoCapture('../Yunting.avi')


# you can also read ay other viddos. 

# list of video frames
frames = []

while(cap.isOpened()):
    # read frame from the video
    ret, frame = cap.read()
    
    if ret is False:
        break
        
    frames.append(frame)

cap.release()

# scale frame to 0-1
frames = np.array(frames) / 255.
print("frames size:", frames.shape, "# (nb_frames, height, width, channel)")

# get height, width
numFrames = frames.shape[0]
height = frames.shape[1]
width = frames.shape[2]
# print(frames, numFrames, height, width)

"""Question  1a : display space-time slice of the video. Figure 1.1 in the chapter. 

Import a short video, and create a 2D plot where Y axis is $t$ and X-axis is $n$, which is the **horizontal** cross secxtion of the movie. See lecture 11, Slide 22. 
Hint: at each frame, take a vector of pixels at a fixed y-position and show an image of nFrames*nhorizontal pixels as a final image. 

References:
- https://stackoverflow.com/questions/65827830/disabledfunctionerror-cv2-imshow-is-disabled-in-colab-because-it-causes-jupy
- https://stackoverflow.com/questions/28962502/python-2-7-and-opencv-code-gives-cvtcolor-error
- https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
- https://stackoverflow.com/questions/61775424/flipped-video-capture-and-display-in-mac-webcam-with-opencv-and-pycharm


"""

finalFrameLst=[]
for i in range(height):
  tmpList = []
  for j in range(numFrames):
    tmp_frame = frames[j]
    sliceFrame = np.array([])
    sliceFrame = tmp_frame[i,:,:]
    tmpList.append(sliceFrame)
  finalFrameLst.append(tmpList)
finalFrameLst = np.array(finalFrameLst)

print(len(finalFrameLst))
print(finalFrameLst[0].shape)
print(type(finalFrameLst[0]))
print(finalFrameLst[0].shape)
# cap.release()
plt.imshow(finalFrameLst[25]) # random select the 25th
plt.xlabel("n"); plt.ylabel("t")

"""import numpy as np
import cv2

cap = cv2.VideoCapture('xxxx.avi')

while (cap.isOpened()):
  ret, frame = cap.read()

  # if frame is successfully read is True
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2_imshow(gray)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print(type(gray))

Question 1b: Create a space-temporal Gaussian filter. Figure 1.3 in the cvBookTemporal.pdf. See Slide 28. 
Gaussian temporal filtering (applied on a sequence of images) will blur the sequence evolution, smoothing out the temporal variation, like a rapid variation in illumination or movement of an object. It's a gaussian filtering of the signal obtained by the temporal evolution of each single pixel.

### Reference:
- https://www.geeksforgeeks.org/how-to-generate-2-d-gaussian-array-using-numpy/
- http://cs229.stanford.edu/section/gaussians.pdf (Multivariate Gaussian Distribution)
"""

# generating synthetic spatio-temporal data with Gaussian process

# Initializing value of x-axis and y-axis from range -5 to 5
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
d = np.sqrt(x*x + y*y)


# Intializing sigma and muu
sigma = 3
mu = 0


# Calculating Gaussian array
gauss = np.exp(-1/2*np.transpose((d - mu)) * sigma**-1 * (d - mu))


plt.imshow(gauss, cmap= "gray")

"""Question 2: Motion Magnification.

Question 2a.
"""

def imshow(im, cmap='gray'):
    # clip image from 0-1
    im = np.clip(im, 0, 1)
    plt.imshow(im, cmap=cmap)

# 9x9 images
imSize = 9

# we would like to magnify the change between im1 and im2 by 4x
magnificationFactor = 4;

# horizontal movement from (0, 0) to (0, 1)
im1 = np.zeros([imSize, imSize])
im2 = np.zeros([imSize, imSize])
im1[0,0] = 1
im2[0,1] = 1

ff1 = fftshift(fft2(im1))
ff2 = fftshift(fft2(im2))

"""##Magnify Change

Firstly, we compute the prase shift between img2 to img 1 in the Fourier transform domain as `phaseShift`. Second, we utilize the position of the origin image to time the exponential of `magnificationFactor` x `phaseShift` x complex number, which magnifies the phase change in frequency domain (`magnifiedDft`). Thirdly, we inverse `magnifiedDft` and keep the real part to get the magnified img as `magnified`.

"""

print(angle(fft2(im2)))

def magnifyChange(im1, im2, magnificationFactor):
    
    # find phase shift in frequency domain
    im1Dft = fft2(im1)
    im2Dft = fft2(im2)
    phaseShift = angle(im2Dft) - angle(im1Dft) # TODO
    
    # magnify the phase change in frequency domain
    magnifiedDft = im1Dft * np.exp(magnificationFactor * phaseShift * 1j) # TODO
    
    # what does the magnified phase change cause in image space?
    magnified = np.fft.ifft2(magnifiedDft).real; 
    
    return magnified

"""HINT: If you're not familiar with complex number in python, here's a quickstart."""

# create a complex number
x = 1 + 1j
print("x =", x)
print("x.real", x.real, "x.imag", x.imag)

# magnitude and phase of complex number
mag = abs(x)
phase = angle(x) # use `angle` to see the phase

print("Magnitude", mag)
print("Phase", phase)

# Euler's formula
y = mag * exp(phase * 1j)
print("y =", y,  y.real, y.imag)

# magnify position change
magnified = magnifyChange(im1, im2, magnificationFactor);

plt.figure(figsize=(12,36))
plt.subplot(131)
plt.imshow(im1); plt.title('im1');

plt.subplot(132)
plt.imshow(im2); plt.title('im2');

plt.subplot(133)
plt.imshow(magnified); plt.title('magnified');

"""##Problem 3.b

We now have two phases, meaning that we must move in two directions (vertical and horizontal). Because the `magnifyChange` function can only be applied to one phase and cannot determine which phase should be moved. Thus, multidirectional phases cannot be properly magnified. Please see the plot below.
"""

# 9x9 images
imSize = 9

# we would like to magnify the change between im1 and im2 by 4x
magnificationFactor = 4

# horizontal movement from (0, 0) to (0, 1)
# additional vertical movement from (8, 8) to (7, 8)
im1 = np.zeros([imSize, imSize])
im2 = np.zeros([imSize, imSize])
im1[0,0] = 1
im2[0,1] = 1
im1[8,8] = 1
im2[7,8] = 1

# magnify position change
magnified = magnifyChange(im1, im2, magnificationFactor)


plt.figure(figsize=(12,36))
plt.subplot(131)
plt.imshow(im1); plt.title('im1');

plt.subplot(132)
plt.imshow(im2); plt.title('im2');

plt.subplot(133)
plt.imshow(magnified); plt.title('magnified');

"""##Problem 3.c

How can we solve the problem if the motion is multidirectional? If there are several movements between two images, one solution is to use a **localized Fourier transform**, which involves individually magnifying the offsets on small windows of the images and aggregating the effects across the windows. When we narrow our scope of consideration, everything in it is more likely to move in the same direction.  
"""

# 9x9 images
imSize = 9

# we would like to magnify the change between im1 and im2 by 4x
magnificationFactor = 4

# width of our Gaussian window
sigma = 2

# horizontal movement from (0, 0) to (0, 1)
# additional vertical movement from (8, 8) to (7, 8)
im1 = np.zeros([imSize, imSize])
im2 = np.zeros([imSize, imSize])
im1[0,0] = 1
im2[0,1] = 1
im1[8,8] = 1
im2[7,8] = 1

# we will magnify windows of the image and aggregate the results
magnified = np.zeros([imSize, imSize])
# magnified = np.zeros(imSize)

# meshgrid for computing Gaussian window
[X, Y] = np.meshgrid(np.arange(imSize), np.arange(imSize))
print(X)

for y in range(0, imSize, 2*sigma):
  for x in range(0, imSize, 2*sigma):
    gaussianMask = np.exp(-((X-x)**2 + (Y-y)**2) / (2*sigma**2)) # TODO
    gaussianIm1 = im1 * gaussianMask # TODO
    gaussianIm2 = im2 * gaussianMask # TODO
    windowMagnified = magnifyChange(gaussianIm1, gaussianIm2, magnificationFactor) # TODO
    magnified = magnified + windowMagnified
        
plt.figure(figsize=(12,36))
plt.subplot(131)
imshow(im1); plt.title('im1');

plt.subplot(132)
imshow(im2); plt.title('im2');

plt.subplot(133)
imshow(magnified); plt.title('magnified');

"""##Problem 3.d"""

import numpy as np
import cv2

cap = cv2.VideoCapture('../Yunting.avi')

# list of video frames
frames = []

while(cap.isOpened()):
    # read frame from the video
    ret, frame = cap.read()
    
    if ret is False:
        break
        
    frames.append(frame)

cap.release()

# scale frame to 0-1
frames = np.array(frames) / 255.
print("frames size:", frames.shape, "# (nb_frames, height, width, channel)")

# get height, width
numFrames = frames.shape[0]
height = frames.shape[1]
width = frames.shape[2]

print(numFrames, height, width)

"""##Motion magnification
Fill out code here marked with #TODO
"""

# 10x magnification of motion
magnificationFactor = 50 # default 10

# width of Gaussian window
sigma = 13 # default 13

# alpha for moving average
alpha = 0.5

# we will magnify windows of the video and aggregate the results
magnified = np.zeros_like(frames)

# meshgrid for computing Gaussian window
X, Y = np.meshgrid(np.arange(width), np.arange(height))

# iterate over windows of the frames
xRange = list(range(0, width, 2*sigma))
yRange = list(range(0, height, 2*sigma))
numWindows = len(xRange) * len(yRange)
windowIndex = 1

for y in yRange:
    for x in xRange:
        for channelIndex in range(3): # RGB channels
            for frameIndex in range(numFrames):
                
                # create windowed frames
                gaussianMask = np.exp(-((X-x)**2+(Y-y)**2) / (2*sigma**2)); # TODO
                windowedFrames = gaussianMask * frames[frameIndex,:,:,channelIndex]
            
                # initialize moving average of phase for current window/channel
                if frameIndex == 0:
                    windowAveragePhase = angle(fft2(windowedFrames))
                
                windowDft = fft2(windowedFrames) 
                
                # compute phase shift and constrain to [-pi, pi] since
                # angle space wraps around
                windowPhaseShift = angle(windowDft) - windowAveragePhase
                windowPhaseShift[windowPhaseShift > pi] = windowPhaseShift[windowPhaseShift > pi] - 2 * pi
                windowPhaseShift[windowPhaseShift < -pi] = windowPhaseShift[windowPhaseShift < -pi] + 2 * pi
                
                # magnify phase shift
                windowMagnifiedPhase = magnificationFactor * windowPhaseShift  # TODO
                 
                # go back to image space
                windowMagnifiedDft = windowDft * np.exp(windowMagnifiedPhase*1j) # TODO
                windowMagnified = abs(ifft2(windowMagnifiedDft))
                
                # update moving average
                windowPhaseUnwrapped = windowAveragePhase + windowPhaseShift
                windowAveragePhase = alpha * windowAveragePhase + (1 - alpha) * windowPhaseUnwrapped
                
                # aggregate
                magnified[frameIndex,:,:,channelIndex] = magnified[frameIndex,:,:,channelIndex] + windowMagnified
        
        # print progress
        print('{}/{}'.format(windowIndex, numWindows))
        windowIndex += 1

outputs = magnified / np.max(magnified)
for channelIndex in range(3):
    originalFrame = frames[0,:,:,channelIndex]
    magnifiedFrame = outputs[0,:,:,channelIndex]
    scale = np.std(originalFrame[:]) / np.std(magnifiedFrame[:])
    originalMean = np.mean(originalFrame[:])
    magnifiedMean = np.mean(magnifiedFrame[:])
    outputs[:,:,:,channelIndex] = magnifiedMean + scale * (outputs[:,:,:,channelIndex] - magnifiedMean)

outputs = np.clip(outputs, 0, 1)

# create output video
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Yunting_magnified50.avi' ,fourcc, 30.0, (height, width))

for i in range(frames.shape[0]):
    # scale the frame back to 0-255
    frame = (np.clip(outputs[i], 0, 1) * 255).astype(np.uint8)
    
    # write frame to output video
    out.write(frame)

out.release()

# Only for colab downloading videos
try:
    from google.colab import files
    files.download('Yunting_magnified50.avi')
except:
    print("Only for google colab")

"""See the HZ (frame/second) for magnified video"""

from moviepy.editor import VideoFileClip

clip = VideoFileClip('Yunting_magnified50.avi')
cap = cv2.VideoCapture('Yunting_magnified50.avi')

# list of video frames
frames = []

while(cap.isOpened()):
    # read frame from the video
    ret, frame = cap.read()
    
    if ret is False:
        break
        
    frames.append(frame)

cap.release()

# scale frame to 0-1
frames = np.array(frames) / 255.
print("frames size:", frames.shape, "# (nb_frames, height, width, channel)")

# get height, width
numFrames = frames.shape[0]
height = frames.shape[1]
width = frames.shape[2]

print(numFrames, height, width)
print("HZ is", numFrames/ clip.duration)