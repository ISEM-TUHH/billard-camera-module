# Run this script with sudo privileges to register the cron job and start the server

service=billard-camera-server

# relative to this file:
#templatefolder=../../templates/

echo "Installing ${service}..."

echo "Creating virtual python environment '.venv'..."
python -m venv ../.venv # create venv
source ../.venv/bin/activate # activate venv

echo "Installing required (simple) python packages into .venv..."
pip install -r requirements.txt # just the simple stuff

# install specific version of ultralytics[export]
echo "Now installing ultralytics[export]==8.3 package. Fixed to this version, as higher version cause error during install."
pip install ultralytics[export]==8.3 && echo "Successfully installed ultralytics[export]!"


mkdir fragments && cd fragments # just a general collection of installation quirks

# build and install cv2 
# possible problem: in my original documentation, I do so outside the venv?
echo "Now to the long part: Building and installing OpenCV (opencv-python) from source with gstreamer support. This can take a long time!"
git clone --recursive https://github.com/opencv/opencv-python.git
cd opencv-python
export CMAKE_ARGS="-DWITH_GSTREAMER=ON" # interesting would also be Cuda support? I think something like that should exist.
pip wheel . --verbose -w # -w should put it into cwd? https://pip.pypa.io/en/stable/cli/pip_wheel/
# how to find wheel location?
pip install $(ls opencv-python*.whl)
cd .. # return to fragments

# install jetcam
git clone https://github.com/NVIDIA-AI-IOT/jetcam
cd jetcam
sudo python3 setup.py install

# add download_page.html as symbolik link to the template folder of 
#ln -s ${templates}/download_page.html download_page.html

echo "Creating systemd service..."
cp ./job.service /etc/systemd/system/${service}.service && echo "File succesfully moved"

systemctl daemon reload && echo "Systemctl daemon reloaded"
systemctl start ${service}.service && echo "Service succesfully started"
systemctl enable ${service}.service && echo "Service succesfully registered for startup"

systemctl status ${service}.service

echo "IP addresses:"
ip a | grep inet
