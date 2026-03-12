from billard_base_module.Module import Module
from billard_base_module.RemoteModules import Beamer
from BallDetector import BallDetector
from GameImage import GameImage

from VideoCapture import VideoCapture

from flask import Flask, render_template, Response, jsonify, request
from werkzeug.exceptions import ClientDisconnected
import requests
import cv2
#from picamera2 import Picamera2
import time
#import apriltag
import numpy as np
import pandas as pd
import datetime
from timeit import default_timer as timer
import json
import os
import signal
import uuid

import vpi
from jetcam.csi_camera import CSICamera

class Camera(Module):
	"""Camera module for the Billard@ISEM system.

	"""

	videoStreaming = False #: tracks if there is a current videostream
	lastVideoFrame = 0 #: the latest frame that was generated
	latestFrameTime = 0 #: timestamp of the latest generated frame
	lastPing = 0 #: latest ping from a liveline call (timestamp)
	recalibrate = False #: state to track if there is a call to recalibrate the image to the ArUco markers
	zoomout = False #: state to track if there is a call to zoom out the camera view (remove calibration)
	counterPictures = 0 #: the number of images saved with incremental names

	lastPositions = None #: last measured coordinates

	# parameters for detection of the corners of the field using apriltags
	#options = apriltag.DetectorOptions(families="tag36h11")
	#detector = apriltag.Detector(options)
	#ah, aw = int(1520*1.5), int(2028*1.5) #: maximum resolution for apriltags detection -> copy of full image gets scaled down to this

	# ArUco markers to replace apriltags
	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) #: cv2 ArUco dictionary of 4x4 family
	detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters()) #: cv2 ArUco detector

	ph, pw = 1115, 2230 #: 9ft pool table measurements 223x111.5cm 
	h, w = 3040, 4032 #: height and width of the picamera image
	

	def __init__(self, config="config/config.json", test_config="config/test_config.json", template_folder="templates"):
		"""Setup the instance.

		This defines the API interface (see api_dict in source).

		"""
		current_dir = os.path.dirname(__file__)
		Module.__init__(self, config=os.path.join(current_dir, config), test_config=os.path.join(current_dir, test_config), template_folder=os.path.join(current_dir, template_folder), static_folder=os.path.join(current_dir, "static"))

		self.beamer = Beamer(self.getModuleConfig("beamer"))
		self.do_beamer_calibration = False

		
		self.worker_uuid = None #: this tracks which uuid (client connection) actually generates the frame, all other connections only listen and send to cached frames. This can dynamically change through Camera.reassign_worker_gen. If None the stream will end.
		self.all_uuids = []

		#if not self.TEST_MODE:
		#	import vpi # this library is only accessible on the Jetson, which is not my testing local environment.

		# Camera initialisation
		#self.picam2 = Picamera2()
		#camera_config = self.picam2.create_still_configuration(main={"size": (self.w, self.h)}, lores={"size": (640, 480)}, display="lores")
		#self.picam2.configure(camera_config)
		#self.picam2.start()
		self.do_quick_inference = False
		self.quick_detector = BallDetector(mode="8pool-quick", debug=False, correction=self.config["3D-position"])

		# Camera with OpenCV
		# available for PiCam HQ: 4032x3040@21, 3840x2160@30, 1920x1080@60
		height, width, fps = 3040, 4032, 21
		#self.cam = CSICamera(width=self.w, height=self.h, capture_width=width, capture_height=height, capture_fps=fps)
		self.cam = VideoCapture(width=self.w, height=self.h, capture_width=width, capture_height=height, capture_fps=fps)
		self.white_balance_reset = 105 # after how many frame the white balance scaler should be reset: 5s@21FPS
		self.white_balance_counter = self.white_balance_reset # the number of frames since the last reset -> trigger instantly on the first call
		self.white_balance_timings = {"normal": [], "mean": []}



		# BallDetector init -> load all YOLO Models
		self.ballDetector = BallDetector(mode="8pool-detail", debug=False, correction=self.config["3D-position"])

		# init self.M as saved matrix (from config/transformation-matrix.json)
		self.load_matrix()

		# load the distortion correction
		self.load_distortion_correction()

		# Get the current amount of taken images, to give every image another name
		with open("config/counter.txt", "r") as file:
			self.counterPictures = int(file.readline())

		self.end_stream = False
		api_dict = {
			"v1": {
				"coords": self.get_coords,
				"image": self.get_image,
				"cacheimage": self.cache_image,
				"savepic": self.do_savepic,
				"zoomout": self.do_zoomout,
				"calibrate": self.do_calibrate,
				"gameimage": self.get_game_image,
				"startbeamercalibration": self.start_beamer_calibration,
				"loadtransformation": self.load_matrix,
				"savetransformation": self.save_matrix,

				"togglequick": self.toggle_quick,
				"stopgeneration": self.stop_generation
			},
			"website": {
				"liveline": self.liveline,
				"video_feed": self.video_feed,
			},
			"meta": {
				"relaunch": self.force_restart
			},
			"": self.index
		} #: The definition of the module specific API endpoints

		#self.detect = YOLO(...)
		#self.classify = YOLO(...)
		self.add_all_api(api_dict)

	def index(self):
		"""Renders and returns the index.html website"""
		#self.stop_generation() # stop generating frames
		print(f"Client connected.")
		return render_template('index.html')

	def load_matrix(self):
		""" Load the transformation matrix from transformation-matrix.json

		If images are being generated and not zoomed out, this transformation is applied to them.

		Returns:
			str: Message containing the matrix
		"""
		self.recalibrate = False
		self.zoomout = False
		with open("config/transformation-matrix.json", "r") as file:
			transTotal = json.load(file)
			self.M = np.array(transTotal)
			#print(M, M.dtype)
		return f"Loaded matrix {self.M}"

	def save_matrix(self):
		"""Save the current transformation

		The current matrix Camera.M gets written to `config/transformation-matrix.json`.

		Returns:
			str: Message containing the matrix

		"""
		with open("config/transformation-matrix.json", "w") as file:
			#print(M, M.dtype)
			asStr = json.dumps(self.M.tolist(), indent=4)
			file.seek(0)
			file.write(asStr)
			file.truncate()
		return f"Written matrix {self.M}"

	def load_distortion_correction(self):
		"""Load the distortion correction from storage.

		Load the Camera.[mtx, dist, rvecs, tvecs] from `config/distortion_correction.json`.
		These should be calculated by `vpi_camera_calibration_analysis_polynomial.py` using `cv2.calibrateCamera`.

		"""
		with open("config/distortion_correction.json", "r") as file:
			kd = json.load(file)
			self.mtx = np.array(kd["mtx"])
			self.dist = np.array(kd["dist"]).flatten()
			self.rvecs = np.array(kd["rvecs"]).flatten()
			self.tvecs = np.array(kd["tvecs"]).flatten()[0:2]

			#print(M, M.dtype)
		return f"Loaded matrix {self.M}"

	def get_coords(self, image=None):
		"""Detect billiard balls in the current frame
		
		The current frame (gathered using `Camera.get_image_internal`) and passes it to the `Camera.ballDetector.detect`.
		
		Returns:
			Respone: flask response containing json of the coordinates in real dimensions.

		"""
		if image is None:
			image = self.get_image_internal()

		self.cached_image = image
		detections = self.ballDetector.detect(image, plot=False)
		realPositions = self.ballDetector.toRealDim(detections, (self.pw,self.ph))
		
		# control
		#self.ballDetector.verify(image, detections)

		self.lastPositions = realPositions
		return jsonify(realPositions)

	def get_game_image(self):
		"""Get a GameImage of the current ball positions

		This is a debug tool to quickly check if corrections are working correctly.
		The GameImage used here is simplified compared to the implementation used in the game module (see https://github.com/ISEM-TUHH/billard-game-module/blob/main/Game/GameImage.py).

		Returns:
			Response: flask response of mimetype `image/jpg` 

		"""
		if self.lastPositions == None: # if this has not been called yet
			self.get_coords()
		pos = self.lastPositions
		gameImage = GameImage()
		gameImage.placeAllBalls(pos)
		img = gameImage.getImageCV2()

		_, buffer = cv2.imencode(".jpg", img)
		return Response(buffer.tobytes(), mimetype="image/jpg")

	def cache_image(self):
		"""Caches the current frame

		This is used for later on saving the current image for training purposes if the coordinates where corrected.

		Returns:
			str: Message that the image was cached
		"""
		self.cached_image = self.get_image_internal()
		return "Cached current image"

	def get_image(self):
		"""Get the current frame
		
		Returns:
			Response: flask response of mimetype `image/jpg`
		"""
		image = self.get_image_internal()
		
		_, buffer = cv2.imencode(".jpg", image)
		return Response(buffer.tobytes(), mimetype="image/jpg")

	def liveline(self):
		"""Updates the Camera.lastPing to the current time

		This is an old mechanic to notice disconnects. 
		This is not needed anymore since commit b9696d4e92e815e172aa00cb59d3a2ef3cbe6717 added closing the camera generation/stream on client disconnect.
		It still exists as some legacy websites hit up this endpoint.

		Returns:
			str: "tiptop"
		
		"""
		self.lastPing = timer()
		return "tiptop"
	
	def do_calibrate(self):
		"""Raise the flags for the calibration procedure

		Raises Camera.recalibrate and Camera.zoomout, which cause the Camera.gen method to zoomout and look for new ArUco markers.

		Returns:
			Response: rendered `calibration_tutorial.html` to guide the user through the calibration. 
		"""
		self.recalibrate = True
		self.zoomout = True
		return render_template("calibration_tutorial.html")

	def do_zoomout(self):
		"""Raise the flag to zoomout

		This causes the camera stream to zoomout.

		Returns:
			str: "tiptop"
		"""
		self.zoomout = True
		self.recalibrate = False # prevent it from zooming in again and crashing due to apriltag errors.
		return "tiptop"

	def do_savepic(self):
		"""Takes the last taken image from the buffer (self.lastVideoFrame) and writes them to ./images/ as .png
		If json data has the tag "action": "save-labels", look for coordinates in the data and save them in the YOLO label format.

		:return: jsonified name of the image.
		"""
		#global counterPictures, frameFin, frame
		#frame = picam2.capture_file(f"training_images/image-{counterPictures}.jpg")
		data = request.json
		print(data)
		filename = f"image-{self.counterPictures}"
		folder = "images"

		if data["action"] == "save-labels":
			coords = data["coords"]
			folder = "images/training"
			h, w = self.config["table-dimensions"]["height"], self.config["table-dimensions"]["width"]

			table = []
			for ball in coords.values():
				table.append({
					"class": ball["name"],
					"x": ball["x"] / w,
					"y": ball["y"] / h,
					"width": self.config["relative-size"]["width"],
					"height": self.config["relative-size"]["height"]
				})

			df = pd.DataFrame(table)
			df.to_csv(f"{folder}/{filename}-rough.txt", sep=" ", header=None, index=False)

			img = self.cached_image
		else:
			img = self.get_image_internal()
		cv2.imwrite(f"{folder}/{filename}.png", img)
		self.counterPictures += 1
		with open("config/counter.txt", "w") as file:
			file.write(f"{self.counterPictures}")
		return jsonify({"answer": f"Last image name: image-{self.counterPictures-1}.jpg"})

	def toggle_quick(self):
		"""Toggles Camera.do_quick_inference
		
		This causes Camera.gen to always pass the image to `Camera.quickDetector.detect` and send the detections directly to the beamer module.

		Returns:
			str: Message wether the quick inference has been enabled or disabled 
		"""
		self.do_quick_inference = not self.do_quick_inference
		return "Enabled quick inference" if self.do_quick_inference else "Disabled quick inference"

	def force_restart(self):
		""" This function kills the process (stops the server).
		
		This should restart the server, as it is listed in systemctl with restart=always
		
		Returns:
			str: "Restarting the server." (this should never reach the client, as the server terminates before)
		"""
		os.kill(os.getpid(), signal.SIGINT)
		return "Restarting the server."

	###### Organization of streaming threads ###################################

	def video_feed(self):
		"""Stream the camera to the client

		To prevent generating multiple frames for multiple clients, each stream is assigned a uuid.
		This allows the system to track that only one thread generates images, with all the others only receiving cached frames.
		When closing the Response, the uuid is removed from the set of active streams (`Camera.stop_my_stream`)

		Returns:
			Response: flask response of mimetype `multipart/x-mixed-replace; boundary=frame`
		"""
		own_uuid = uuid.uuid1() # this generates a unique identifier for this generation thread
		self.all_uuids.append(own_uuid) # register with organizer

		#if self.worker_uuid == None:
		#	self.reassign_worker_gen()
		# 
		self.worker_uuid = own_uuid # the must current stream always gets assigned as the active generator
		#self.all_uuids = [own_uuid]
		response = Response(self.gen(own_uuid=own_uuid), mimetype='multipart/x-mixed-replace; boundary=frame')
		response.call_on_close(lambda: self.stop_my_stream(own_uuid))
		return response

	def stop_generation(self):
		"""Forcefully stop the camera stream

		Returns:
			str: "Deactivated stream"

		"""
		self.end_stream = True
		self.cam.stop_stream(force=True)
		self.videoStreaming = False
		print("Ended stream.")
		return "Deactivated stream"

	def stop_my_stream(self, uuid):
		"""Stop a stream thread by removing the uuid.
		
		Also reassigns the worker uuid to the most current one. 
		
		Args:
			uuid (uuid.UUID): the uuid to remove

		Returns:
			uuid.UUID: the passed uuid
		"""
		self.all_uuids.remove(uuid)
		self.reassign_worker_gen()
		self.cam.stop_stream()
		return uuid

	def reassign_worker_gen(self):
		""" This function looks at all registered uuid (client connection) and selects the next connection that should be the worker. The newest one is chosen. 
		
		Returns:
			uuid.UUID: the worker thread uuid
		
		"""
		if len(self.all_uuids) > 0:
			self.worker_uuid = max(self.all_uuids)
		else:
			self.worker_uuid = None

		print("Worker thread:", self.worker_uuid)
		print("All threads:", self.all_uuids)
		return self.worker_uuid

	###### Beamer Module interactions ##########################################

	def send_quick_inference(self, frame):
		"""Detect balls and send the coordinates in simple format to the Beamer Module.

		Simple format is defined as {"points": [{"x": 123, "y": 123}, {...}, ...]}
		"""
		points = np.array(self.quick_detector.detect(frame, plot=False)["results"])
		if len(points) > 0 or True: # always push to beamer
			color = (0,0,0)
			#print("QUICK INFERENCE, n_objects=", len(points))
			for point in points:
				cv2.drawMarker(frame, point, color=color, markerType=cv2.MARKER_CROSS)
				#cv2.putText(img, f"{r['name']}: {r['conf']:.2f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)
			requests.post(self.beamer.endpoint("/v1/dynamicballs"), json={"points": [{"x": int(x[0]), "y": int(x[1])} for x in points]}, headers={"content-type": "application/json"})
		return

	def start_beamer_calibration(self):
		"""When this is hit (http GET), starts looking for the beamer markers in Camera.gen (must be running)

		Camera.gen calls Camera.beamer_calibration on every loop, see that method for details. This endpoint gets hit up by the beamer module.
		"""
		self.do_beamer_calibration = True
		return f"Now looking for ArUco markers with the IDs {self.config['aruco-id-beamer']}"

	def beamer_calibration(self, frame):
		"""Looks for the beamer tags in the frame.

		If all four tags are found, sends the coordinates to the beamer module.

		Returns the annotated frame, annotates found markers.
		"""
		goal_ids = self.config["aruco-ids-beamer"] # ids of apriltags projected by the beamer onto the table ordered top-left, top-right, bottom-left, bottom-right
		#img = self.get_image_internal()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#print("BEAMER CALIBRATION: looking for ArUco IDs", goal_ids)

		corners, ids, rejected = self.detector.detectMarkers(gray)
		if ids is not None:
			frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

			src_points = np.zeros((4,2))
			points = 0
			for tagCorners, tagId in zip(corners, ids):
				tagId = int(tagId[0])
				print(tagId, type(tagId), goal_ids, type(goal_ids[0]))
				if tagId in goal_ids:

					centerPoint = np.sum(tagCorners[0], axis=0)/4
					print("Beamer TagId:", tagId)
					src_points[tagId - min(goal_ids), :] = np.float32(centerPoint)
					points += 1

				if points == 4: # as the recalibration is finished if there are 4 corners entered into src_points
					self.do_beamer_calibration = False
					# generate a GET argument string like "?x1=123&y1=123&x2=...&y4=123" to send to the beamer
					query = "?" + "&".join([f"x{i+1}={r[0]}&y{i+1}={r[1]}" for i, r in enumerate(src_points)])
					print("BEAMER CALIBRATION: sending query to BM", query)
					requests.get(self.beamer.endpoint("/v1/camin") + query)
					#self.stop_generation() # stop the livestream
		return frame

	###### background functions ################################################

	def white_balance(self, vpi_image, stats_image):
		"""Performes a color correction based on the grey world algorithm on a provided image using the VPI API.

		Performance benchmark on a 1115x2230 BGR8 image (Jetson Orin Nano Super): without mean ~1ms, with mean ~65ms on a 4032x3040 iamge. Discussions can be held on the frequency of resetting the white balance. The default is every 105 frames (5s).

		Args:
			vpi_image (vpi.Image): image to get corrected. Should be BGRA8 format, BGR8 should also work.
			stats_image (vpi.Image): image on which the scaling should be calculated. This seperation is useful if only a small portion of this image actually gets used further on, but the entire image is needed for white balance

		Returns:
			vpi.Image: the color corrected image

		"""
		start = timer()
		if self.white_balance_counter >= self.white_balance_reset:
			vpi_stats = stats_image.image_stats(flags=vpi.ImageStatistics.MEAN) # only calculates the mean of each channel. Not adding this flag causes it to do more stats, doubling the computational time (roughly 80ms)
			
			channel_means = vpi_stats.cpu().view(np.recarray)[0][0][:3] # the first entry (index 0) are the means of each channel

			#time_stats = timer()

			gray_mean = np.mean(channel_means) # ignore alpha channel
			
			self.white_balance_channel_scales = gray_mean / channel_means
			self.white_balance_counter = 0
		self.white_balance_counter += 1

		channels = [
			vpi.Image(vpi_image.size, vpi.Format.U8),
            vpi.Image(vpi_image.size, vpi.Format.U8),
            vpi.Image(vpi_image.size, vpi.Format.U8)
		]
		vpi.mixchannels([vpi_image], channels, [0,1,2], [0,1,2]) # see https://docs.nvidia.com/vpi/algo_mix_channels.html

		# now remap each channel separately
		scaled_channels = []
		for scale, channel in zip(self.white_balance_channel_scales, channels):
			scaled_channels.append(channel.convert(vpi.Format.BGR8, scale=scale))			

		output = vpi.Image(vpi_image.size, vpi.Format.BGR8)

		# for unknown reasons, the U8 channels also seem to have three channels? This could cause memory and performance hits, but it works.
		vpi.mixchannels(scaled_channels, [output], [0,3,6], [0,1,2])

		time = (timer() - start)*1000 # ms
		#print("WHITE BALANCE took", np.round(time, 2), "ms")

		#self.white_balance_timings["normal" if self.white_balance_counter != 1 else "mean"].append(time)

		# clear memory: often needed in VPI...
		for channel, schannel in zip(channels, scaled_channels):
			del channel
			del schannel

		return output

	def get_image_internal(self):
		"""Retrieve the current image from the camera

		Either uses the latest frame from the livestream or generates a single new frame.

		Returns:
			np.ndarray: cv2 image
		"""
		start = timer()
		image = 0
		print(f"videoStreaming: {self.videoStreaming}")
		if self.videoStreaming:
			print("Grabbing an already generated image")
			image = self.lastVideoFrame.copy()
		else:
			print("Generating a new image")
			# as self.gen always returns a generator object, it has to be iterated over, even if it just writes to object values.
			for i in self.gen(once=True):
				continue
			#return (b'--frame\r\n'
			#		b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg", self.lastVideoFrame)[1].tobytes() + b'\r\n')
			image = self.lastVideoFrame.copy()

		print(f"get_image_internal took {(timer()-start):.3f} s")
		return image

	def gen(self, once=False, own_uuid=None): # not in api directly
		"""Generate an image (stream) and yield on every generation. Prevent double generation from different clients by writing to self.lastVideoFrame and returning that instead of generating an entire new image.

		THIS RETURNS A GENERATOR OBJECT (due to yield, even if they are not reached in the structure). To actually execute this outside of video_feed, put it in a "for i in self.gen(...): continue". Only this actually calls the functions :) 

		:param once: take just one image (run through the entire method once)
		:type once: optional bool

		"""

		self.videoStreaming = True
		hasBeenCalibrated = False

		# runtime optimisation: just use this once and then undistort the frame using cv2.remap(...) insted of cv2.undistort()
		#map1, map2 = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (self.w,self.h), cv2.CV_32FC1) # TODO: experiment what changes with CV_16FC1?

		# use Nvidia VPI to setup lens correction: vpi.WarpGrid
		if not self.TEST_MODE:
			grid = vpi.WarpGrid((self.w, self.h)) # setup dense grid
			#self.mtx = np.array([[1.46775111e+03, 0.00000000e+00, 2.00867103e+03], [0.00000000e+00, 1.47483400e+03, 1.50325726e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

			K = self.mtx[0:2, :] # loaded from distortion_correction.json
			X = np.eye(3,4)
			#dist = np.array([0.32255492,  0.00360983,  0.00209136, -0.01833096]) # dist = np.array([[-0.54452965,  0.32255492,  0.00360983,  0.00209136, -0.01833096]])
			#dist = np.array([0.02110592, 0.07828829, 0.00089954, 0.0465791])

			#print("DIST:", self.dist, self.dist.shape)
			warp = vpi.WarpMap.polynomial_correction(grid, Kin=K, X=X, Kout=K,# dist=self.dist,
                                      rcoeffs=self.dist)#, tcoeffs=self.tvecs) # https://docs.nvidia.com/vpi/python/build/vpi.WarpMap.html



		#dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # TODO: put in the measurements of the pool table
		dst_points = np.float32([[0, 0], [self.pw, 0], [0, self.ph], [self.pw, self.ph]]) # gets set once
		src_points = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]) # init here, gets assigned every calibration loop
		

		#camera = cv2.VideoCapture(0)
		#while True:
		self.lastPing = timer()
		print("Generating frames")
		#while (timer() - self.lastPing) < 60 or once: # 60 seconds after no new Ping (js: fetch("liveline")), stop generating new frames
		ret = True

		lastFrameTime = 0 # when only listening and forwarding, this checks if there has already been a new frame generated to send.

		self.cam.start_stream() # bufferless stream

		self.end_stream = False
		skip_frame = True
		while not self.end_stream and own_uuid in self.all_uuids: # raising self.end_stream ends all streams, while removing own_uuid from self.all_uuids only kills the associated generator.

			if self.worker_uuid == own_uuid:
				#print("Frame generated through uuid", own_uuid)
				start = timer() # for timing frame generation

				frameRaw = self.cam.read() # jetcam CSICamera object

				if not ret:
					raise AssertionError("Frame was not read properly. Is the device busy?")

				capturing = timer()

				#h,  w = frameRaw.shape[:2]
				#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
				#print(h,w)
				#frameUndistorted = cv2.undistort(frameRaw, mtx, dist, None, newcameramtx) # takes roughly 50% of the entire time to generate image
				# -> roughly 110ms@1520x2028px 
				#frameUndistorted = frameRaw # cv2.remap(frameRaw, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
				# -> combined with initUndistortRectifyMap only roughly 28ms@1520x2028 :D

				undistortion = timer()

				#frame = frameUndistorted # picamera2 gives BGR, cv2 RGB apparently
				# used to be cv2.remap(img, map1, map2, cv.INTER_LINEAR)
				# TODO: undistortion with VPI?
				
				colors = timer()

				#ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

				# init vector for source points of perspective transform
				points = 0 # counter of points/corners entered into src_points

				results = []#
				matrix = self.M
				if self.recalibrate:
					with vpi.Backend.CUDA:
						# this pipeline does: to VPI -> to BGR -> lens correction -> perspective warp
						vpi_raw = vpi.asimage(frameRaw, vpi.Format.BGRA8).convert(vpi.Format.BGR8)
						frameVPI = self.white_balance(vpi_raw.remap(warp), stats_image=vpi_raw)#, border=vpi.Border.MIRROR)
						with frameVPI.rlock_cpu() as data:
							frame = data
						#del frameVPI
						del vpi_raw
							#print("RECALIBRATION SHAPE:", frame.shape)

					# TODO: change back to apriltags detection, this time using VPI. Low priority, as the performance here is not critical.
					grayBig = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
					gray = grayBig # cv2.resize(grayBig, (self.aw, self.ah)) # reduces quality for apriltags detection -> not needed atm for aruco?
					#results = self.detector.detect(gray) # apriltags
					corners, ids, rejected = self.detector.detectMarkers(gray)
					#print("recalibration", corners, ids)
					if ids is not None:
						frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

						#print("ARUCO IDs:", ids)

						for tagCorners, tagId in zip(corners, ids):
							if tagId in [1,2,3,4] and self.recalibrate:

								#centerPoint = np.sum(tagCorners[0], axis=0)/4
								#print(tagId, tagCorners, centerPoint)
								indexCorner = [1, 2, 4, 3][int(tagId) - 1]
								print("TagId:", tagId)
								corner = tagCorners[0, indexCorner - 1] # because of this solution, we must follow the order top left = id1, top right = id2, bottom left = id3 and bottom right = id4 when calibrating
								src_points[tagId - 1, :] = np.float32(corner)
								#src_points[tagId - 1, :] = np.float32(centerPoint)
								points += 1

							if points == 4: # as the recalibration is finished if there are 4 corners entered into src_points
								self.recalibrate = False
								self.zoomout = False
								self.M = cv2.getPerspectiveTransform(src_points, dst_points)

				if not self.zoomout: # only if there are 4 detected tags
					# using CUDA backend on Nvidia Jetson Orin Nano
					if not self.TEST_MODE:
						#print("WARP PERSPECTIVE, frame shape:", frame.shape)
						with vpi.Backend.CUDA:
							# this pipeline does: to VPI -> to BGR -> lens correction -> perspective warp
							vpi_frame = (vpi.asimage(frameRaw, vpi.Format.BGRA8)
								.convert(vpi.Format.BGR8)
							)
							#frameVPI3 = self.white_balance(frameVPI2)
							frameVPI = self.white_balance(
								#vpi.asimage(frameRaw, vpi.Format.BGRA8)
								vpi_frame
								#.convert(vpi.Format.BGR8)
								.remap(warp)
								.perspwarp(self.M)
								.view(vpi.RectangleI(0, 0, self.pw, self.ph)),
								stats_image=vpi_frame
							) # TODO: maybe change to BGR8, but also adapt in jetcam.csi_camera.CSICamera
							#frameVPI = vpi.asimage(frameRaw, vpi.Format.BGRA8).convert(vpi.Format.BGR8).perspwarp(self.M).view(vpi.RectangleI(0, 0, self.pw, self.ph))
							#print("1 FrameVPI sync:", frameVPI)
							with frameVPI.rlock_cpu() as data:
								frame = data
								#print("2 FrameVPI async:", frame.shape, frameVPI)
							#print("3 FrameVPI")
							#print("The frameVPI.rlock_cpu() is asyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyync")
							# delete frameVPI later on, wait for async...
							del vpi_frame
					else: # this should never get used
						#frame = cv2.warpPerspective(frame, self.M, (self.pw, self.ph)) # cols = w, rows = h, TODO: put in the new desired image size also here -> Pool table
						pass

				if self.zoomout and not self.recalibrate:
					# if only zoomed out but no active recalibration: just apply perspwarp
					with vpi.Backend.CUDA:
						# this pipeline does: to VPI -> to BGR -> lens correction -> perspective warp
						vpi_frame = vpi.asimage(frameRaw, vpi.Format.BGRA8).convert(vpi.Format.BGR8)
						frameVPI = self.white_balance(vpi_frame.remap(warp), stats_image=vpi_frame)#.convert(vpi.Format.BGR8).remap(warp)
						with frameVPI.rlock_cpu() as data:
							frame = data
						#del frameVPI
						del vpi_frame

				self.lastVideoFrame = frame.copy()

				if self.do_quick_inference:
					# if we want to do quick inference (just show detected balls on the image)
					self.send_quick_inference(frame)

				if self.do_beamer_calibration:
					frame = self.beamer_calibration(frame)

				end = timer()
				#print(f"Frame created in {end-start}s, capture: {capturing-start}, undistortion: {undistortion-capturing}, coloring: {colors-undistortion}, apriltags/warp: {end-colors}")
				#print(type(frame))

				if once:
					print("Generated a new image using self.gen with once=True")
					self.videoStreaming = False
					self.cam.stop_stream() # only this instance
					return

				# downsize the frame for livestream -> bandwidth problems
				#print("BUFFER SHAPE:", frame.shape)
				with vpi.Backend.CUDA:	
					frameVPI = (vpi.asimage(frame, vpi.Format.BGR8)
						.rescale(np.array(frame.shape)[[1,0]]//3) # TODO: maybe change to BGR8, but also adapt in jetcam.csi_camera.CSICamera
					)
					with frameVPI.rlock_cpu() as data:
						frame = data

				self.lastVideoFrameLowRes = frame
				self.latestFrameTime = end
				del frameVPI

			else: # if this is not the worker thread, just listen and send the self.lastVideoFrameLowRes
				if lastFrameTime != self.latestFrameTime:
					#print("Mirroring frame to uuid", own_uuid)
					lastFrameTime = self.latestFrameTime
					frame = self.lastVideoFrameLowRes
				else:
					continue

			#print("Yielding frame, shape:", frame.shape)
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg", frame)[1].tobytes() + b'\r\n')
				#b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg", cv2.resize(self.lastVideoFrame, (self.w//2, self.h//2)))[1].tobytes()  + b'\r\n')

		if self.worker_uuid == None:
			self.videoStreaming = False
			self.cam.stop_stream(force=True) # this forces the end of the current camera thread. Can be reopened.
			print("Framegen has ended")

		#if own_uuid in self.all_uuids:
		#	self.all_uuids.remove(own_uuid)
		#self.cam.stop_stream() # this just decreases the counter of active instances by 1. If the counter is 0, the camera thread ends.


if __name__ == "__main__":
	cam = Camera(template_folder="templates")
	#cam.add_api(cam.get_coords, "v1/coords")
	#cam.get_image()
	print(os.getpid())

	if cam.TEST_MODE:
		cam.app.run(host="0.0.0.0", port="5002")
	else:
		cam.app.run(host="0.0.0.0", port="5000")

	# Experiment: timings of the white balancing algorithm
	#normal = cam.white_balance_timings["normal"]
	#mean = cam.white_balance_timings["mean"]
	#print("WHITE BALANCE timings stats")
	#print("Normal:", np.mean(normal), "+-", np.std(normal), "@", len(normal), "total entries")
	#print("With mean:", np.mean(mean), "+-", np.std(mean), "@", len(mean), "total entries")