from Module import Module
from BallDetector import BallDetector
from GameImage import GameImage

from flask import Flask, render_template, Response, jsonify, request
import cv2
#from picamera2 import Picamera2
import time
#import apriltag
import numpy as np
import pandas as pd
import datetime
from timeit import default_timer as timer

class Camera(Module):
	"""Camera module for the billard robot.

	THIS IS THE DUMB CAMERA. Only for testing on a device without an available Picamera2. Does not actually generate an image/stream, just loads/sends prerecorded images. 
	"""

	videoStreaming = False # tracks if there is a current videostream
	lastVideoFrame = 0 # 
	latestFrameTime = 0 # timestamp of the last generated frame
	lastPing = 0 # latest ping from a liveline call
	recalibrate = True # state to track if there is a call to recalibrate the image to the fiducials
	zoomout = False # state to track if there is a call to zoom out the camera view (delete calibration)
	counterPictures = 0

	# last measured positions
	lastPositions = None

	# parameters for detection of the corners of the field using apriltags
	#options = apriltag.DetectorOptions(families="tag36h11")
	#detector = apriltag.Detector(options)
	ah, aw = int(1520*1.5), int(2028*1.5) # maximum resolution for apriltags detection -> copy of full image gets scaled down to this

	ph, pw = 1171, 2150# 9ft pool table measurements: 257x127cm 
	h, w = 3040, 4056#1520, 2028 # height and width of the picamera image
	
	dist = np.array([[-0.54452965,  0.32255492,  0.00360983,  0.00209136, -0.01833096]])

	mtx = np.array([[2.07474799e+03, 0.00000000e+00, 1.02664792e+03], [0.00000000e+00, 2.06829116e+03, 7.44089884e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
	scale_x = w/2028
	scale_y = h/1520
	mtx[0,0] *= scale_x
	mtx[1,1] *= scale_y
	mtx[0,2] *= scale_x
	mtx[1,2] *= scale_y

	def __init__(self, config="config/config.json", template_folder=""):
		Module.__init__(self, config=config, template_folder=template_folder)
		
		# Camera initialisation
		"""self.picam2 = Picamera2()
		camera_config = self.picam2.create_still_configuration(main={"size": (self.w, self.h)}, lores={"size": (640, 480)}, display="lores")
		self.picam2.configure(camera_config)
		self.picam2.start()"""

		# BallDetector init -> load all YOLO Models
		self.ballDetector = BallDetector(mode="8pool-detail")

		# Get the current amount of taken images, to give every image another name
		with open("config/counter.txt", "r") as file:
			self.counterPictures = int(file.readline())

		api_dict = {
			"v1": {
				"coords": self.get_coords,
				"image": self.get_image,
				"savepic": self.do_savepic,
				"zoomout": self.do_zoomout,
				"lenscorrection": self.do_lenscorrection,
				"calibrate": self.do_calibrate,
				"gameimage": self.get_game_image
			},
			"website": {
				"liveline": self.liveline,
				"video_feed": self.video_feed,
			},
			"": self.index
		}

		#self.detect = YOLO(...)
		#self.classify = YOLO(...)
		self.add_all_api(api_dict)

	def index(self):
		print(f"Client connected.")
		return render_template('index.html')

	def get_coords(self):
		"""Takes a picture of the pool table an determines the postion of each ball in the common frame of reference.
		Returns a dict of all detected balls and the time of today in seconds (float).
		"""
		stamp = time.time()
		image = self.get_image_internal()
		detections = self.ballDetector.detect(image)
		realPositions = self.ballDetector.toRealDim(detections, (self.pw,self.ph))
		
		# control
		self.ballDetector.verify(image, detections)

		self.lastPositions = realPositions
		return jsonify(realPositions)

	def get_game_image(self):
		if self.lastPositions == None: # if this has not been called yet
			self.get_coords()
		pos = self.lastPositions
		#print(pos)
		gameImage = GameImage()
		gameImage.placeAllBalls(pos)
		img = gameImage.getImageCV2()

		_, buffer = cv2.imencode(".jpg", img)
		return Response(buffer.tobytes(), mimetype="image/jpg")


	def get_image(self):
		image = self.get_image_internal()
		
		_, buffer = cv2.imencode(".jpg", image)
		return Response(buffer.tobytes(), mimetype="image/jpg")

	def video_feed(self):
		image = self.get_image_internal()
		
		_, buffer = cv2.imencode(".jpg", image)
		return Response(buffer.tobytes(), mimetype="image/jpg")
		#return Response(self.gen(self.picam2), mimetype='multipart/x-mixed-replace; boundary=frame')

	def liveline(self):
		self.lastPing = timer()
		return "tiptop"
	
	def do_calibrate(self):
		self.recalibrate = True
		return "tiptop"

	def do_zoomout(self):
		self.zoomout = True
		return "tiptop"

	def do_lenscorrection(self):
		#self.lenscorrection = True
		# does nothing yet 
		return "tiptop"

	def do_savepic(self):
		"""Takes the last taken image from the buffer (self.lastVideoFrame) and writes them to ./images/ as .png

		:return: jsonified name of the image.
		"""
		#global counterPictures, frameFin, frame
		#frame = picam2.capture_file(f"training_images/image-{counterPictures}.jpg")
		cv2.imwrite(f"images/image-{self.counterPictures}.png", self.lastVideoFrame)
		self.counterPictures += 1
		with open("config/counter.txt", "w") as file:
			file.write(f"{self.counterPictures}")
		return jsonify({"answer": f"Last image name: image-{self.counterPictures-1}.jpg"})




	###### background functions ################################################

	def get_image_internal(self):
		image = 0
		return cv2.imread("images/image-73.png")

		print(f"videoStreaming: {self.videoStreaming}")
		if self.videoStreaming:
			print("Grabbing an already generated image")
			image = self.lastVideoFrame
		else:
			print("Generating a new image")
			#self.gen(...)
			for i in self.gen(self.picam2, once=True):
				continue
			#return (b'--frame\r\n'
			#		b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg", self.lastVideoFrame)[1].tobytes() + b'\r\n')
			image = self.lastVideoFrame
		return image

	def gen(self, camera, once=False): # not in api directly
		"""Generate an image (stream) and yield on every generation. Prevent double generation from different clients by writing to self.lastVideoFrame and returning that instead of generating an entire new image.

		THIS RETURNS A GENERATOR OBJECT (due to yield, even if they are not reached in the structure). To actually execute this outside of video_feed, put it in a "for i in self.gen(...): continue". Only this actually calls the functions :) 

		:param once: take just one image (run through the entire method once)
		:type once: optional bool

		"""
		try:			
			if self.videoStreaming and not once:
				print("Forwarding frames")
				lastFrameTime = self.latestFrameTime
				while (timer() - self.lastPing) < 60: # listen to updates of lastVideoFrame and if there was no ping in the last 60s, stop generating/sending frames
					if lastFrameTime == self.latestFrameTime:
						continue
					# TODO: time this to see how often more we are sending frames than generating new ones to optimise runtime
					lastFrameTime = self.latestFrameTime
					yield (b'--frame\r\n'
						b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg", self.lastVideoFrame)[1].tobytes() + b'\r\n')
				print("Forwarding ended")

			
			hasBeenCalibrated = False

			# scaling down for apriltags -> higher resolution causes crashes :/
			scaleFactor = self.w/self.aw # scaling between full image an scaled down version for apriltags
			#h, w = 1520, 2028 # -> now defined on top of the file
			newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w,self.h), 1, (self.w,self.h)) # yes this can change image size in the second (w,h) pair, but not good for cropping image -> see
			# hast been originally calculated for (2028,1520) image size.

			# runtime optimisation: just use this once and then undistort the frame using cv2.remap(...) insted of cv2.undistort()
			map1, map2 = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (self.w,self.h), cv2.CV_32FC1) # TODO: experiment what changes with CV_16FC1?


			#dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # TODO: put in the measurements of the pool table
			dst_points = np.float32([[0, 0], [self.pw, 0], [0, self.ph], [self.pw, self.ph]]) # gets set once
			src_points = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]) # init here, gets assigned every calibration loop
			

			#camera = cv2.VideoCapture(0)
			#while True:
			self.lastPing = timer()
			print("Generating frames")
			while (timer() - self.lastPing) < 60 or once: # 60 seconds after no new Ping (js: fetch("liveline")), stop generating new frames
				#print(timer()-lastPing)
				self.videoStreaming = True

				start = timer() # for timing frame generation

				frameRaw = self.picam2.capture_array()

				capturing = timer()

				#h,  w = frameRaw.shape[:2]
				#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
				#print(h,w)
				#frameUndistorted = cv2.undistort(frameRaw, mtx, dist, None, newcameramtx) # takes roughly 50% of the entire time to generate image
				# -> roughly 110ms@1520x2028px 
				frameUndistorted = cv2.remap(frameRaw, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
				# -> combined with initUndistortRectifyMap only roughly 28ms@1520x2028 :D

				undistortion = timer()

				frame = cv2.cvtColor(frameUndistorted, cv2.COLOR_BGR2RGB)
				grayBig = cv2.cvtColor(frameUndistorted, cv2.COLOR_BGR2GRAY)
				gray = cv2.resize(grayBig, (self.aw, self.ah)) # reduces quality for apriltags detection
				
				colors = timer()

				#ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

				# init vector for source points of perspective transform
				points = 0 # counter of points/corners entered into src_points

				results = []
				if self.recalibrate:
					results = self.detector.detect(gray) # apriltags
					for r in results:
						# extract the bounding box (x, y)-coordinates for the AprilTag
						# and convert each of the (x, y)-coordinate pairs to integers
						(ptA, ptB, ptC, ptD) = r.corners
						ptB = (int(scaleFactor*ptB[0]), int(scaleFactor*ptB[1]))
						ptC = (int(scaleFactor*ptC[0]), int(scaleFactor*ptC[1]))
						ptD = (int(scaleFactor*ptD[0]), int(scaleFactor*ptD[1]))
						ptA = (int(scaleFactor*ptA[0]), int(scaleFactor*ptA[1]))
						# draw the bounding box of the AprilTag detection -> no time for this shit
						cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
						cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
						cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
						cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
						# draw the center (x, y)-coordinates of the AprilTag
						(cX, cY) = (int(scaleFactor*r.center[0]), int(scaleFactor*r.center[1]))
						cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
						# draw the tag id on the image
						tagId = r.tag_id
						cv2.putText(frame, str(tagId), (ptA[0], ptA[1] - 15),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

						# put into perspective transform
						if tagId < 4 and tagId >=0 and self.recalibrate:
							#Cundistorted = cv2.undistortPoints(np.array([cX, cY]), newcameramtx, dist)
							src_points[tagId, :] = np.float32([cX,cY])#Cundistorted#np.float32([cXundist,cYundist])
							points += 1
							self.zoomout = False
						if points == 4: # as the recalibration is finished if there are 4 corners entered into src_points
							self.recalibrate = False

				if len(results) == 4 or not self.recalibrate: # only if there are 4 detected tags
					#rows, cols, _ = frame.shape
					# definition of dst_points pulled out of loop, on top

					if not self.zoomout:
						matrix = cv2.getPerspectiveTransform(src_points, dst_points) # TODO: pull this out of the loop? -> benchmark timing first to determine necessity

						# Wende die Transformation an
						frame = cv2.warpPerspective(frame, matrix, (self.pw, self.ph)) # cols = w, rows = h, TODO: put in the new desired image size also here -> Pool table
				self.lastVideoFrame = frame

				end = timer()
				self.latestFrameTime = end
				#print(f"Frame created in {end-start}s, capture: {capturing-start}, undistortion: {undistortion-capturing}, coloring: {colors-undistortion}, apriltags/warp: {end-colors}")
				#print(type(frame))

				if once:
					print("Generated a new image using self.gen with once=True")
					self.videoStreaming = False
					return

				yield (b'--frame\r\n'
					b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg", frame)[1].tobytes() + b'\r\n')
			self.videoStreaming = False # on closing the website this is never reached
			print("Framegen has ended") 
		except Exception as e:
			print(e)


if __name__ == "__main__":
	cam = Camera(template_folder="templates")
	#cam.add_api(cam.get_coords, "v1/coords")
	#cam.get_image()
	print(cam.api_flat)
	cam.app.run(host="0.0.0.0")