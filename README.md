# Camera Module for Billard@ISEM
Implementation of a computer vision system for the Billard@ISEM system.

This module provides functionality for image calibration and an API for livestreaming, infering billiard ball coordinates/numbers and much more.

## Hardware setup
- Currently using a Nvidia Jetson Orin Nano with a Raspberry HQ Camera. 
    - Any other Nvidia Jetson should do, as we rely on their VPI library for image processing. Older commits (before the branch `jetson-port`) work on any computer by using CPU resources instead of hardware accelerations, which is mostly slower.
- The current lens correction setup is only for lenses where a polynomial lens correction is applicable (most normal lenses). If your camera setup uses a lens, for which a fisheye model is more applicable, you would have to modify some parts of the source code.

## Installation
The installation process for this module is quite complex, since it relies on a lot of optimisation steps. The installation can take multiple hours, depending on your hardware.
1. Clone this repository: `git clone --recurse-submodules https://github.com/ISEM-TUHH/billard-camera-module.git`
    - Downloading into the `/home/<user>` directory simplifies the next steps.
2. Edit `config/config.json` to have the correct IP-addresses of your other modules.
3. Move into the installation directory: `cd billard-camera-module/install` 
4. Edit `billiard-camera-server.service`: correct the user from `isem` to whatever the user name on your system is. Also correct the paths to the executable/directory, if you did not clone into the home directory.
5. To start the main installation process, run `sudo chmod +x ./install.sh && sudo ./install.sh`
    - This will make the install script executable and run it.
    - The script will create a virtual python environment (called `.venv`) and install the dependencies
        - This includes building `opencv-python` from source to be compatible with cameras connected via CSI. From experience, this takes a long time.
    - A new `systemd` service (in `/etc/systemd/system/`) called `billard-camera-module.service` will be created, enabled but not started.
        - This is the main service that should run when to system is used. Starting it is shown later on in this document.
    - A root cronjob to reboot the system every night at 4:00am will be created. From our experience, this solves a lot of problems.
    - If you do not want some of these steps, you can comment them out in the `install.sh` file.
6. From the root directory of the repo, run `./launch.sh` to test the system. It should start the camera with the configuration from `config/config.json`, avaialable under the port described in that file.
    - The video shown in the browser is very likely weird, as you still need to calibrate the camera.
    - If you have a camera with manual focus and already mounted your camera in the desired position, you can use the livestream to set the focus.

## Calibration
The calibration processes are supported by phsyical objects, mostly certain visual patterns. See `resources/` for the exact objects we used. 

### Lens correction
Unless your camera comes pre corrected, you'll need to correct for lens distortion. If you want more details on lens distortion, read up for example [here](https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html).
1. Have a printout of a calibration checkerboard pattern available.
    - From our experience, you'll get the best results if you do this after mounting and focusing the camera in the final position (see 6. from Installation).
    - Checkerboard patterns can be generated as .pdf [here (third party)](https://calib.io/pages/camera-calibration-pattern-generator). This is only tested for "Target Type" "Checkerboard".
    - The checkerboard size should be sufficient to be well visible in the camera. We used an A0 print at approx. 3m distance.
    - Place the printed pattern on some rigid surface. There should be no curves or creases.
2. Navigate to the root of the repo
3. Activate the virtual environment: `source .venv/bin/activate`
4. Capture calibration images using `python vpi_camera_calibration_capture.py`
    - If your camera has a different framerate to 21, edit this in the file before.
    - This script captures a total of 15 images, one every 8 seconds
    - Between each capture, move the checkerboard around to have maximum coverage of the image. Also rotate the pattern.
    - There now should be 15 new images in `config/calibration_images`
    - You can repeat this as often as you want until you think your images are good enough.
5. Run the calibration script: `python vpi_camera_calibration_analysis_polynomial.py`
    - If your checkerboard pattern has a different size than 15x21 rows/columns, change the `CHECKERBOARD` variable in the script. It should contain the number of borders between rows/columns.
    - This script takes all the `.png` images in directory `config/calibration_iamges` and uses them to calculate the best fitting polynomial lens correction parameters
    - Results are saved into `config/distortion_correction.json` and an examplary corrected image in `config/calibrated_images`.
6. The next time the camera software starts, it will use the saved lens correction parameters.

### Image cropping
We want to crop the image to the corners of the billard table, so that the camera infers correct coordinates. If you also use the [Beamer Module](https://github.com/ISEM-TUHH/billard-beamer-module), you should calibrate them onto the same corners. This must be done with the Camera Module software running, as it is done dynamically.
1. Create a set of markers to mark the corners.
    - Print 4 ArUcO markers of the dictionary/family 4x4, the IDs 1, 2, 3 and 4. We use an edge length of approx. 76mm.
    - When printing and cutting them out, be aware that a small white border around the marker is essential. Also keep track of which marker is which.
    - Create a fixture of some kind to reproducibly place the markers in the corners of the table. A lasercut template is available in the `resources/` directory.
        - With our solution: by pushing the fixture into the corner, place the corresponding marker onto the fixture such that the corner is directly beneath the edges of the billard table.
2. Open the website of the module
3. Open the link `/v1/calibrate` in a new tab (background)
4. Now, the livestream video should be zoomed out and looking for the four markers. 
    - As soon as all four are found, the image crops onto them.
    - Found markers are drawn with a (thin) green line around them
    - If your camera has problems finding them, we found the following debugging steps to be helpful:
        - With a manual focus camera, check all corners are in focus
        - Change the lighting
        - Consider printing larger markers
5. If they are zoomed in correctly, you can now click on `v1/savetransformation` from the modules homepage to save it.
    - In the future, this configuration will be used.
6. You are done and can now remove the markers. This calibration only needs to be repeated when you change something.

![Alignment of the first ArUco marker in the top-left corner](https://github.com/ISEM-TUHH/billard-camera-module/blob/jetson-port/docs/source/images/marker-alignment.jpeg?raw=true)
*Alignment of the first ArUco marker in the top-left corner*

## Getting images
Every now and then, you might want to download/extract all files that can be used in training. Use `zip images.zip images/*/*` to zip all images in subfolders of the `images` directory.