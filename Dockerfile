# Builds a docker image for deployment on a Nvidia Jetson board
# We use a Nvidia Jetson Orin Nano Dev Kit (8 GB) with Jetpack 6.x installed

FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /app

COPY requirements.txt .

RUN apt-get update
#RUN apt-get install -y python3-pip python3-dev && \
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# End of cached

COPY . .

EXPOSE 5000

CMD ["python3", "Camera.py"]

# run using
# sudo docker run --runtime=nvidia --device=/dev/video0 -p 5000:5000 ma4096/billard_camera:latest