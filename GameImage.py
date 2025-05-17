import numpy as np
#import cv2
from PIL import Image, ImageDraw, ImageFont

class GameImage:
	"""Class to generate an image from different game modes and other information

	Each image starts with a black background, as this is "neutral" on a beamer.
	:param size: width x height of the image in pixels
	:type size: tuple<int>
	:param phys: how many meters are 1000 pixels? (equivalent to how many mm are 1 pixel)
	:type phys: float
	"""
	ballDiameter = 50.8 # diameter of an 8 pool ball in mm

	def __init__(self, size=(2150, 1171), phys=1):
		self.img = Image.new(mode="RGB", size=size, color="#50b12c")
		self.draw = ImageDraw.Draw(self.img)
		self.phys = phys
		self.w, self.h = size

	def getImageCV2(self):
		return np.array(self.img)[:,:,[2,1,0]] # shift from rgb to bgr

	def placeBall(self, pos, n, d=None):
		"""Place a ball on the canvas self.img based on its number

		:param pos: (x,y) center of the ball in mm from the top left
		:type pos: tuple<int>
		:param n: number of the ball
		:type n: int
		:param d: diameter of the ball in pixels, automatically determined to be accurate according to self.phys and 
		:type d: optional int
		"""
		if d == None:
			d = int(self.ballDiameter/self.phys)
			print(d)
		
		b = BilliardBall(n)
		bImg = b.getImg(d)
		x,y = tuple([int(i/self.phys - d//2) for i in pos])
		print(n, ": ",x,y, pos)
		self.img.paste(bImg, (x,y), bImg) # second call for mask, so the corners dont get overwritten

	def placeAllBalls(self, data):
		"""Place all balls mentioned in the data dict on the canvas self.img

		:param data: dictionary matching the number of a ball (key) to its position (tuple x,y) from the upper left in mm
		:type data: dict<tuple<float>> | list<dict(name(str), x, y)>
		"""
		for b in data:
			if type(data) == list:
				rawN = b["name"]
				number = 8 if rawN=="eight" else (16 if rawN=="white" else int(rawN))
				x,y = b["x"],b["y"]
				self.placeBall((x,y),number)
				continue
			self.placeBall(data[b], int(b))

	def instructionText(self, text):
		"""Place text on top and flipped on the bottom of the image. Font size is chosen automatically.

		:param text: Text to be placed
		:type text: str
		"""
		fs = int(self.h/15)# if self.w/len(text) < # TODO: make it more dynamic 
		font = ImageFont.truetype("static/Roboto-Black.ttf", fs)

		img = Image.new(mode="RGBA", size=(self.w, int(1.5*fs)), color="#00000000")
		draw = ImageDraw.Draw(img)

		_,_, w, h = draw.textbbox((0,0), text, font = font)
		draw.text(((self.w-w)//2, 0), text, font=font, fill="white", background="#00000000")
		self.img.paste(img, (0,0), img)
		timg = img.transpose(Image.ROTATE_180)
		self.img.paste(timg, (0,int(self.h-1.5*fs)), timg)

	def drawArrow(self, start, end, color="white"):
		"""Draws an arrow from the start to the end (in mm)

		:param start, end: coordinates in mm (x,y)
		:type start, end: tuple<float>
		"""
		startP = tuple([int(i/self.phys) for i in start])
		endP = tuple([int(i/self.phys) for i in end])
		self.draw.line([startP,endP], fill=color, width=0)


class BilliardBall:
	"""Class to store each Billiard Balls graphical attributes

	Each ball is initiated with its number with number 16 being the white ball. Number 1-7 are full, 8 is black, 9-15 are half.
	"""
	colors = ["#ffda07ff","#0046a7","#df0100","#52008b","#e05400","#017902","#740000","#20201e","#ffda07","#0046a7","#df0100","#52008b","#e05400","#017902","#740000","#d5d1c5"]

	def __init__(self, n):
		self.n = n if n != 16 else "" # empty string for the white ball
		self.type = "half" if (n>8 and n<16) else "full"
		if n == 8:
			self.type = "eight"
		elif n == 16:
			self.type = "white"

		self.color = self.colors[n-1]
		self.white = self.colors[-1]


	def getImg(self, d):
		"""Get the image of the ball with its number, half or full and color.
		
		:param d: Diameter of the returned image (side of square) in px
		:type d: int
		:return: PIL.Image object
		"""
		self.font = ImageFont.truetype("static/Roboto-Black.ttf", d//3)
		
		img = Image.new(mode="RGBA", size=(d, d))
		draw = ImageDraw.Draw(img)

		draw.ellipse((0,0,d,d), fill=self.color, outline=self.color) # main color 
		if self.type == "half":
			draw.rectangle((0,0,d,d//5), fill=self.white, outline=self.white)
			draw.rectangle((0,4*d//5,d,d), fill=self.white, outline=self.white)
			draw.ellipse((-d//2,-d//2,3*d//2,3*d//2), outline="#00000000", width=d//2)

		draw.ellipse((d//4,d//4, 3*d//4, 3*d//4), fill=self.white, outline=self.white) # inner white circle
		_,_, w, h = draw.textbbox((0,d//20), str(self.n), font = self.font)
		draw.text(((d-w)//2, (d-h)//2), str(self.n), font=self.font, fill="black")

		return img

if __name__ == "__main__":
	data = {"1": (400,300), "2": (654,76), "16": (1200,900), "14": (1800,400), "8": (700,300), "15": (588,833)}
	g = GameImage()#phys=1, size=(3840,2160))
	g.instructionText("Platziere die Kugeln auf den Projektionen")
	g.placeAllBalls(data)
	#g.drawArrow((100,100),(300,600))

	g.img.show()
	