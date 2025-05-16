import numpy as numpy
import pandas as pd
import os
from flask import Flask, jsonify, render_template
import socket
#import urllib.request
import requests

class Module:
	"""Parent class to modules
	Defines a web interface for modules consisting of HTML website and REST API


	:param id: str to identify the module, can be accessed via API "http://XXX.XXX.XXX.XXX/id"
	:type id: str
	:param template_folder:
	:type template_folder: str, optional 
	"""
	def __init__(self, id, template_folder=""):
		self.api = {"id": (lambda: id)}
		self.app = Flask(__name__, template_folder=template_folder) # template folder would otherwise be "/template/" for stuff like index.html
		self.app.add_url_rule("/id", "id", lambda: jsonify({"id": id}))
		self.app.add_url_rule("/api-doc", "api-doc", lambda: jsonify({"api": self.api_flat}))

		self.api_flat = ["/id"] # list of all api endpoints as full strings
		self.available_modules = None # change via scan_network method, initialized as None to signal it hasn't scanned yet
		return

	def add_website(self, file):
		"""Add a path to a html website/file to be made available under https://xxx.xxx.xxx.xxx/index

		file: absolute path to html file (e.g. index.html)
		"""
		self.index = file

		# return the website
		with self.app.app_context():
			self.app.add_url_rule("/index", "index", lambda: render_template(file))
		return

	def add_api(self, method, path):
		"""add an api point to the service which returns JSON when called

		method: method of the class which should be called when this API is called. Must return a dict object, which gets parsed to JSON and returned on request.
		path: str like "v1/coords" under where the methods output is available. Call like "http://XXX.XXX.XXX.XXX/v1/coords"

		return: no return value
		"""
		self.api_flat.append(path)

		# traverse api paths to find where to put it, check if parenting paths already exist
		path_list = path.split("/")
		cur = {path_list[-1]: method}
		for p in path_list[0:-2:-1]: # reverse from the back and build up deep nested dict
			cur = {p: cur}

		self.api = self.api | cur # merge the two dictionaries

		# set as path on the server
		#self.app.add_url_rule("/" + path, path, lambda: jsonify(method()))
		self.app.add_url_rule("/" + path, path, method)

	def add_all_api(self, api):
		"""Just define a nested dictionary with paths building on each nest, ending on a method. Calls Module.add_api(...) on each path/method.

		:param api: a dictionary like {"v1":{"coords": getCoords}, "v2":{"image": getImage}} gets made available under /v1/coords and /v2/image with the return of each method (getCoords and getImage) getting send as parsed json as a response.
		:type api: <dict<dict...<method>..>>
		"""
		self.recursive_add_all_api(api, "")

	def recursive_add_all_api(self, api, path0):
		for k in api.keys():
			nextPath0 = f"{path0}/{k}"
			nextElement = api[k]
			if type(nextElement) == dict:
				self.recursive_add_all_api(nextElement, nextPath0)
			else:
				self.add_api(nextElement, nextPath0)

	def scan_network(self, ip_range="192.168.0.X"):
		"""Scan the network for other modules and save their ids

		Sets Module.available_modules: dict with id - IP pair.

		ip_range: (optional) str, give the range of IPs to scan (IPv4). Variable parts to be given with "X", iterated from 0 to 255. Default is "192.168.1.X". CURRENTLY NOT USED
		"""

		# get own host ip address
		host = socket.gethostbyname(socket.gethostname())
		print(f"Address of the host: {host}")
		
		# always only iterate the last of the 4 bytes: a.b.c.XXX
		baseIP = ".".join(host.split(".")[:-1]) + "."

		# on localhost
		#baseIP = "127.0.0."

		self.available_modules = {}
		for i in range(0,255):
			# maybe like this? https://stackoverflow.com/questions/51001483/python-list-all-devices-on-local-network-along-with-ip-address
			ip = baseIP + str(i) # from 0 to 255
			#print(ip)
			try:
				con = requests.get(f"http://{ip}:5000/id").json()
				print(f"Found module: {con['id']} on {ip}")
				
				#print(con)
				self.available_modules[con["id"]] = ip
			except Exception:
				print(f"Did not find anything on {ip}")
				1+1			

		print(self.available_modules)
		return

	def modules_available(self, modules):
		"""Check if all modules needed are available

		:param modules: List of module ids/names. Always lowercase!
		:type modules: list(str)
		"""
		ava = self.available_modules.keys()
		missing = []
		for m in modules:
			if m not in ava:
				missing.append(m)
		
		if len(missing) != 0:
			print(f"The following modules where missing when checking available modules for a certain function: {', '.join(missing)}")
			return False

		return True




if __name__ == "__main__":
	#print(Module.__doc__)
	#print(Module.add_api.__doc__)
	mod = Module("Camera")
	mod.add_api(lambda: "1234", "v1/coords")
	mod.add_api(lambda: "14", "v1/pic") 
	#print(mod.api["id"]())
	#print(mod.api)
	mod.add_website("base/index.html")
	# to run the actual api server
	mod.scan_network()
	mod.app.run()
