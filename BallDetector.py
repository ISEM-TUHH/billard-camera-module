from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
from timeit import default_timer as timer
import datetime
import os # only for testing on different machines -> test if a path to dummy data exists
import traceback
from PIL import Image
import matplotlib.pyplot as plt

class BallDetector():
    """Class implementing advanced detection algorithms and plausability checks for a given image and a selected gamemode

    :param mode: select a gamemode (and detection accuracy/level). Available are currently 8pool-simple and 8pool-detail (simple only differentiating half and full balls, detail every single balls number).
    :type mode: optional string
    :param debug: decide if debug statements (like timings etc) should be printed
    :param debug: optional bool        
    """

    classes_simple = ["ball"] # currently exactly like classes_ballRough
    classes_ballRough = ["ball"]

    def __init__(self, mode="8pool-simple", debug=True):
        self.mode = mode
        self.debug = debug if type(debug) == bool else True

        # a dummy image is used to trigger a detection event on the loaded model so it gets placed in memory, lowering response time in the real inference
        dummy_image = np.random.randint(0, 255, (480,640,3))
        match mode:
            case "8pool-quick":
                self.model = YOLO("models/ballPosition.engine", task="detect")
                self.model(dummy_image, verbose=False)
            case "8pool-simple":
                self.model = YOLO("models/best_ncnn_model", task="detect")
            case "8pool-detail":
                self.detectionModel = YOLO("models/ballPosition.engine", task="detect")
                self.detailModel = YOLO("models/detailModel-old.pt", task="classify")
                #self.detectionModel = YOLO("models/ballPosition_ncnn_model", task="detect")
                #self.detailModel = YOLO("models/detailModel_ncnn_model", task="classify") # wrong results

    def detect(self, img, plausability=True, plot=True, img_name=False):
        """Detect pool balls on an image based on the mode selected on init

        :param img: cv2 image object (or link to an image readable by YOLO)
        :type img: cv2 image object
        :param plausability: if True, the result can only have each class once. If False, the same ball class can be detected multiple times
        :type plausability: bool    
        :param plot: if True, a plot with the images and confidence matrix is generated for debugging
        :type plot: bool
        :param img_name: If the img will get saved because of debug decisions, it will get the current datetime as a default name. For batch processing this can be problematic, so the name can also be directly set.
        :type img_name: str, bool

        :return: dict<list of dict(name, x, y, conf), dict<gamemode(str)> of coordinates of detected balls, and used gamemode
        """
        self.h, self.w, _ = img.shape # save for transforming to real dimensions
        self.img = img
        self.img_name = img_name # if not False, this will be the name if the image gets saved for debugging

        startTime = timer()
        output = []
        outputAlt = {}

        if plot:
            fig, ax = plt.subplots(1,2, figsize=(10,5), dpi=150)
            ax[0].axis("off")
            ax[1].axis("off")

        # if there appear to be errors in the inference process, the image should get saved so we can later on train on it and improve the model over time.
        debugSaveFolder = False # this gets set to the subfolder in images/ that matches the error case. If False, the image will not be saved

        try:
            match self.mode:
                case "8pool-quick":
                    outputAlt = []
                    # only detect the balls, no classification
                    results = self.model(img, verbose=self.debug, save=False, exist_ok=True, conf=0.4, iou=0.4)
                    for r in results:
                        boxes = r.boxes
                        if self.debug: print(f"There where {len(boxes)} balls in this result of {len(results)} total results detected.")
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                            xm, ym = int((x1+x2)/2), int((y1+y2)/2) # only use the center
                            outputAlt.append([xm, ym]) # this can directly be parsed into an np.array in the output
                        #print(outputAlt, img.shape)


                case "8pool-simple": # this mode is not very well tested nor does it currently have the right model on the PI
                    results = self.model(img, verbose=self.debug, save=False, exist_ok=True)
                    for r in results:
                        boxes = r.boxes
                        if self.debug: print(f"There where {len(boxes)} balls in this result of {len(results)} total results detected.")
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                            xm, ym = int((x1+x2)/2), int((y1+y2)/2) # only use the center
                        
                            balltype = self.classes_simple[int(box.cls[0])]
                            output.append({"name": balltype, "x": xm, "y": ym})
                            outputAlt[balltype] = {"name": balltype, "x": xm, "y": ym}
                            #if self.debug: print(f"Detected a {balltype} at (middle) x={xm} and y={ym}.")
            
                case "8pool-detail":              
                    # the main ball detection model call. Only gets saved when this is in debug mode (self.debug=True). 
                    # iou and conf are values to cut of the model when it is detecting "bad" stuff (like grouping multiple balls), see https://docs.ultralytics.com/de/modes/predict/#inference-arguments
                    results = self.detectionModel(img, verbose=self.debug, save=self.debug, exist_ok=True, iou=0.4, show_conf=True, show_labels=True, conf=0.4)
                    

                    if plot:
                        image_path = "runs/detect/predict/image0.jpg"
                        ax[0].imshow(Image.open(image_path))
                        #ax[0].imshow(img)

                    for r in results: # loops just once as we only have one result-object. Loops multiple times if we are infering on multiple images above (as done below with the detailModel)
                        boxes = r.boxes
                        if self.debug: print(f"There where {len(boxes)} balls in this result of {len(results)} total results detected.")
                        
                        # these exception already get handled below in `if counterAreas == 0:`
                        #if len(boxes) == 0:
                            # no balls were detected
                            #print("WARNING in BallDetector: no balls were detected! Saved image to images/no_ball.png for control")
                            #debugSaveFolder = "wrong_number_balls"
                            #cv2.imwrite("images/no_balls.png", img)

                        # prevent detecting the same 
                        #class_exists = [False]*16 # currently hardcoded for 16 type of balls ------------------------------------------------------------------
                        confmat = [] # build confidence matrix -> gets parsed to np.array later
                        temp_pos = [] # temporary storage of positions while the plausable class is determined
                        classes = [] # not hardcoding, since ncnn has different order each time (??) -> extract from details result (c.names)

                        # check the area of each box and form the average
                        # if the area of a box is over 1.15x the average area, skip it
                        factor = 1.5
                        sum_area = 0
                        counterAreas = 0
                        if plausability:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                area = (x2-x1)*(y2-y1)
                                if area > 5500:
                                    print(f"Ball at {[x1,y1]} skipped as its area {area} > 5500.") 
                                    continue # just a made up exclusion for 
                                sum_area += area
                                counterAreas += 1

                        if counterAreas == 0:
                            debugSaveFolder = "wrong_number_balls"
                            raise Exception("No balls found.")

                        # avg_area is often around 4000, so rounding it to int does not introduce a significant error. Just looks better when printing to console.
                        avg_area = int(sum_area/counterAreas)

                        if self.debug: print(f"The average area of a box is {avg_area}.")

                        cropped = []
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                            xm, ym = int((x1+x2)/2), int((y1+y2)/2) # only use the center
                            conf = box.conf
                            border = 0 # expand the image in each direction by n pixels

                            # check if the area is significantly over the average area of all boxes (with factor, see above) -> if true, skip this box)
                            if plausability:
                                areaOfBox = (x2-x1)*(y2-y1)
                                upperLimit = avg_area*factor
                                lowerLimit = avg_area/factor
                                #if areaOfBox >= upperLimit or areaOfBox <= lowerLimit or areaOfBox > 5000: # 5k seems a good value
                                if areaOfBox >= 5500 or areaOfBox <= 1000:
                                    debugSaveFolder = "wrong_size_ball"
                                    if self.debug: 
                                        print(f"The box at [{xm},{ym}] is skipped because its area is {areaOfBox} with the cutoff being {min(upperLimit, 5500)}>area>{lowerLimit}. It had a confidence of {box.conf}.")
                                        cv2.imwrite(f"images/cropped/{float(conf)}.png", img[y1-border:y2+border,x1-border:x2+border])
                                    continue

                            temp_pos.append({"x": xm, "y": ym})

                            #print(x1,x2,y1,y2)
                            cropped.append(img[y1-border:y2+border,x1-border:x2+border])

                            # fully deactivated as this uses so many ressources
                            #if self.debug: cv2.imwrite(f"images/cropped/{conf}.png", cropped[-1])

                        if len(cropped) == 0: # prevent trying to infer on no images (-> no detected balls)
                            return {"results": [], "mode": self.mode}

                        # infer all at once to improve timings
                        details = self.detailModel.predict(cropped, save=False, exist_ok=True, verbose=self.debug, imgsz=160) # according to documentation there should be a probs=False option, but YOLO says no :( (https://docs.ultralytics.com/modes/predict/#inference-arguments)

                        classes = np.array(list(details[0].names.values())) # list of all class names ordered like in the model. As far as I know they are always the same for each result, just being alphabetically ordered.
                        for c in details: # like r in results
                            name = c.names[c.probs.top1] # dont take a pre configured names-list, as the model has its own ordered list
                            confOld = float(c.probs.top1conf)
                            #t5 = c.probs.top5
                            probAllClasses = c.probs.data
                            confmat.append(list(probAllClasses.cpu())) # build the confidence matrix row by row
                            
                            #classes = list(c.names.values())
                            if not plausability: # skip checking for only one mention of each class
                                output.append({"name": name, "x": xm, "y": ym, "conf": confOld})
                                outputAlt[name] = {"name": name, "x": xm, "y": ym, "conf": confOld}
                            
                        if plausability:
                            confmat = np.array(confmat)
                            #classes = classesNames
                            #print(classes)
                            temp_pos = np.array(temp_pos)
                            r,c = confmat.shape
                            #print(r,c)
                            
                            # as this is an super important part, there is a lot of data to review when debugging
                            if self.debug and False: # currently disabled 
                                print("Printing confidence matrix:\n", confmat)
                                df = pd.DataFrame(confmat, columns=classes)
                                df.to_csv("confmat.csv", sep="\t")

                                                            
                            if plot:
                                # backup classes and temp_pos for later on:
                                classesBackup = classes
                                temp_posBackup = temp_pos
                                table = ax[1].table(cellText=np.round(confmat,3), colLabels=classes, loc="upper left")
                                #table.auto_set_font_size(False)
                                #table.set_fontsize(12)
                                table.scale(1.0, 1.6)
                                print(fig.bbox.xmax, fig.bbox.ymax)
                                for i,img in enumerate(cropped):
                                    img = cv2.resize(img, (40,40))
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    #print(i)
                                    #table[(i+1,0)].set_text_props(text="")
                                    #table[(i+1,0)].set_facecolor("white")
                                    #fig.figimage(img, fig.bbox.xmax//2+28,fig.bbox.ymax-158-40*i, zorder=1)
                                    fig.figimage(img, fig.bbox.xmax//2+41, fig.bbox.ymax-162-40*i - 20, zorder=1) # aligning the images -40*(9 - len(cropped)) - 20*(len(cropped) % 2)
                                        
                            first_iteration = True
                            while r > 0: # deletes a
                                r,c = confmat.shape if len(confmat.shape)>1 else (1, confmat.shape[0]) # if else to handle 1x1 matrix
                                #print("Shape of confmat: ", r,c)

                                max_in_col = np.argmax(confmat, axis=0)
                                #print(type(max_in_col))
                                if type(max_in_col) != np.ndarray:
                                    #print("int detected")
                                    max_in_col = np.array([int(max_in_col)])

                                # handle 1x1 matrix
                                max_in_row = np.array([0]*r)
                                if r != 1:
                                    max_in_row = np.argmax(confmat, axis=1)
                                elif r == 1: # if confmat is 1xm
                                    max_in_row = np.array([np.argmax(confmat)])

                                    # special handling of column max at 1xm
                                    max_in_col = np.array([0]*c)
                                #print(r,c,max_in_col, confmat.shape)

                                c_axis = np.arange(0,c) # 1,2,3,4,...: matching
                                r_axis = np.arange(0,r)

                                # reorder max_in_row to match the max_in_col columns (ar = aranged)
                                shape = confmat.shape
                                if self.debug: 
                                    print("Shapes of max_in_col and max_in_row: ", max_in_col.shape, max_in_row.shape)
                                    #print(f"max_in_col: {max_in_col}")
                                    #print(f"max_in_row: {max_in_row}")
                                    #print(f"confmat: {confmat}")

                                max_in_row_ar = max_in_row[max_in_col]
                                max_in_col_ar = max_in_col[max_in_row]

                                # dif and map to bool -> if the difference is 0 (match), throw it away in the next step 
                                max_non_match_col = (max_in_row_ar - c_axis) != 0
                                max_non_match_row = (max_in_col_ar - r_axis) != 0 # yes, they are "mixed" up by design!!
                                #print(max_non_match_row, max_non_match_col)
                                #print(max_in_col_ar, r_axis)
                                #iter_classes = classes[~max_non_match_col]
                                iter_rows = max_in_col_ar[~max_non_match_row]
                                
                                for i in iter_rows: # iterate over matched/determined rows and actually add them to the output
                                    pos = temp_pos[i]
                                    #print(max_in_row, i, classes)

                                    # assign the class name
                                    name = classes[max_in_row[i]]

                                    try:
                                        conf = max(confmat[0]) # if 1xm
                                    except: # if it is not iterable
                                        conf = confmat[0] # if 1x1
                                    if r!=1:
                                        conf = confmat[i,max_in_row[i]]

                                    if self.debug: print(f"class: {name}{' '*(8-len(name))} conf: {conf*100:.2f}%")
                                    output.append({"name": str(name), "x": pos["x"], "y": pos["y"], "conf": float(conf)})

                                    outputAlt[name] = {"name": str(name), "x": pos["x"], "y": pos["y"], "conf": float(conf)}

                                    #print(classes[max_in_row_ar[i]], confmat[i, :], max_in_row_ar[i])

                                if self.debug: 
                                    print("Shapes of temp_pos and max_non_match_row: ", temp_pos.shape, max_non_match_row.shape)
                                    #print("max_non_match_row: ", max_non_match_row)
                                temp_pos = temp_pos[max_non_match_row]
                                classes = classes[max_non_match_col]

                                #print("temp_pos:", temp_pos)
                                if len(temp_pos) == 0:
                                    if self.debug: print(f"Classes that have not been populated/registered: {classes}")
                                    break

                                if self.debug: print(f"Reaching another iteration as the class ball(s) on {temp_pos} are not the highest confidence in their top1 classes. Now trying for {classes}")
                                
                                if self.debug: print(confmat.shape, max_non_match_row.shape, max_non_match_col.shape)
                                #confmat = confmat[max_non_match_row, max_non_match_col] # update confmat to new dimensions
                                confmat = np.delete(confmat, np.where(~max_non_match_row)[0], axis=0)
                                confmat = np.delete(confmat, np.where(~max_non_match_row)[0], axis=1)
                                #confmat = confmat[max_non_match_col, max_non_match_row] # update confmat to new dimensions
                                r = confmat.shape[0] # check if there are any remaining rows

                                if r > 0 and first_iteration:
                                    # if there still are rows after the first iteration, that means there are still balls that where not perfectly classifiable.
                                    # in this case, the image should be saved for training later on.
                                    debugSaveFolder = "classification_problems"
                                first_iteration = False # this marks the end of the first iteration

            if plot:
                for i,r in enumerate(output):
                    col = list(classesBackup).index(r["name"])
                    row = list(temp_posBackup).index({"x": r["x"], "y": r["y"]})
                    table[(row+1,col)].set_facecolor("yellow")
                plt.savefig("images/detection.png")

            if self.debug: print(f"Detected objects (total of {len(output)}): \n{outputAlt}\n")
            if self.debug: print(f"Elapsed time for BallDetector.detect: {timer()-startTime}")

            self.saveDebugImage(debugSaveFolder)
            return {"results": outputAlt, "mode": self.mode}
        
        except Exception:
            if self.TEST_MODE:
                print("Caught exception in BallDetector.detect:")
                print(traceback.format_exc())
                self.saveDebugImage(debugSaveFolder)

            return {"results": {}, "mode": "error"}

    def saveDebugImage(self, folder):
        """ If the folder is not False, this will save the BallDetector.img to images/[folder]/[self.img_name or timestamp].png """
        if folder:
            if not self.img_name:
                filename = "{:%Y-%m-%d_%H:%M:%S}.png".format(datetime.datetime.now()) # Maybe one day add subsecond timings if we do faster detections
            else:
                filename = self.img_name
            cv2.imwrite(os.path.join("images", folder, filename), self.img)


    def toRealDim(self, results, dimensionsTable):#(rw,rh)):
        """Returns results["results"] but with x and y as floats 
        """
        rw, rh = tuple(dimensionsTable)
        #trans = []
        #for r in results["results"]:
        #    rx, ry = r["x"]/self.w*rw, r["y"]/self.h*rh
        #    trans.append({"name": r["name"], "x": rx, "y": ry, "conf": float(r["conf"])})
        trans = {}
        for r in results["results"]:
            obj = results["results"][r]
            rx, ry = obj["x"]/self.w*rw, obj["y"]/self.h*rh
            trans[obj["name"]] = {"name": obj["name"], "x": rx, "y": ry, "conf": float(obj["conf"])}

        return trans

    def verify(self, img, results):
        """Overlay an image with its results and save it. IS BROKEN RIGHT NOW
        
        :param image: cv2 image on which the results where inferred using BallDetector.detect
        :param results: output of BallDetector.detect
        """
        output = results["results"]
        for name, r in output.items():
            print(r)
            x,y = r["x"],r["y"]
            color = (0,0,0)
            cv2.drawMarker(img, (x,y), color=color, markerType=cv2.MARKER_CROSS)
            cv2.putText(img, f"{r['name']}: {r['conf']:.2f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

        cv2.imwrite("verifyBallDetection.png", img)



if __name__=="__main__":
    #img = cv2.imread("images/image-73.png")
    img = "images/training/image-1378.png"
    if os.path.exists(img):
        img = cv2.imread(img)
    else:
        print("File not found.")
        exit
    b = BallDetector(debug=True, mode="8pool-detail")
    out = b.detect(img, plot=True)
    b.verify(img, out)
    #b.toRealDim(out, (1000,1000))
    #micrarmat(np.array([3,1,2,1,0]),4,5)
    #micrarmat(np.array([3,1,2,0]),4,5)