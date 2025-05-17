from ultralytics import YOLO
import numpy as np
import cv2
from timeit import default_timer as timer
import os # only for testing on different machines -> test if a path to dummy data exists

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

        match mode:
            case "8pool-simple":
                self.model = YOLO("models/best_ncnn_model", task="detect")
            case "8pool-detail":
                self.detectionModel = YOLO("models/ballPosition.pt", task="detect")
                self.detailModel = YOLO("models/detailModel.pt", task="classify")

    def detect(self, img, plausability=True):
        """Detect pool balls on an image based on the mode selected on init

        :param img: cv2 image object (or link to an image readable by YOLO)
        :type img: cv2 image object
        :param plausability: if True, the result can only have each class once. If False, the same ball class can be detected multiple times
        :type plausability: bool

        :return: dict<list of dict(name, x, y, conf), dict<gamemode(str)> of coordinates of detected balls, and used gamemode
        """
        self.h, self.w, _ = img.shape # save for transforming to real dimensions
        startTime = timer()
        output = []

        match self.mode:
            case "8pool-simple":
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
                        #if self.debug: print(f"Detected a {balltype} at (middle) x={xm} and y={ym}.")
        
            case "8pool-detail":
                results = self.detectionModel(img, verbose=self.debug, save=False, exist_ok=True, iou=0.4, show_conf=False, show_labels=False)
                for r in results:
                    boxes = r.boxes
                    if self.debug: print(f"There where {len(boxes)} balls in this result of {len(results)} total results detected.")
                    
                    # prevent detecting the same 
                    #class_exists = [False]*16 # currently hardcoded for 16 type of balls ------------------------------------------------------------------
                    confmat = [] # build confidence matrix -> gets parsed to np.array later
                    temp_pos = [] # temporary storage of positions while the plausable class is determined
                    classes = [] # not hardcoding, since ncnn has different order each time (??) -> extract from details result (c.names)

                    cropped = []
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                        xm, ym = int((x1+x2)/2), int((y1+y2)/2) # only use the center
                        temp_pos.append({"x": xm, "y": ym})

                        #print(x1,x2,y1,y2)
                        cropped.append(img[y1-10:y2+10,x1-10:x2+10])

                    # infer all at once to improve timings
                    details = self.detailModel.predict(cropped, save=False, exist_ok=True, verbose=False) # according to documentation there should be a probs=False option, but YOLO says no :( (https://docs.ultralytics.com/modes/predict/#inference-arguments)

                    classes = np.array(list(details[0].names.values())) # list of all class names ordered like in the model
                    for c in details: # like r in results
                        name = c.names[c.probs.top1] # dont take a pre configured names-list, as the model has its own ordered list
                        confOld = float(c.probs.top1conf)
                        #t5 = c.probs.top5
                        probAllClasses = c.probs.data
                        confmat.append(list(probAllClasses)) # build the confidence matrix row by row
                        
                        #classes = list(c.names.values())
                        if not plausability: # skip checking for only one mention of each class
                            output.append({"name": name, "x": xm, "y": ym, "conf": confOld})
                        
                    if plausability:
                        confmat = np.array(confmat)
                        #classes = classesNames
                        #print(classes)
                        temp_pos = np.array(temp_pos)
                        r,c = confmat.shape
                        #print(r,c)
                        #print(confmat)
                        while r > 0: # deletes a
                            r,c = confmat.shape if len(confmat.shape)>1 else (1, confmat.shape[0]) # if else to handle 1x1 matrix
                            #print(r,c)

                            max_in_col = np.argmax(confmat, axis=0)
                            #print(type(max_in_col))
                            if type(max_in_col) != np.ndarray:
                                print("int detected")
                                max_in_col = np.array([int(max_in_col)])

                            # handle 1x1 matrix
                            max_in_row = np.array([0]*r)
                            if r != 1:
                                max_in_row = np.argmax(confmat, axis=1)

                            c_axis = np.arange(0,c) # 1,2,3,4,...: matching
                            r_axis = np.arange(0,r)

                            # reorder max_in_row to match the max_in_col columns (ar = aranged)
                            shape = confmat.shape
                            #print(max_in_row)
                            #print(max_in_col)
                            #print(max_in_col, max_in_row)
                            max_in_row_ar = max_in_row[max_in_col]#micrarmat(max_in_col, shape) @ max_in_row #
                            max_in_col_ar = max_in_col[max_in_row]#micrarmat(max_in_row, shape) @ max_in_col

                            # dif and map to bool -> if the difference is 0 (match), throw it away in the next step 
                            max_non_match_col = (max_in_row_ar - c_axis) != 0
                            max_non_match_row = (max_in_col_ar - r_axis) != 0 # yes, they are "mixed" up by design!!

                            #iter_classes = classes[~max_non_match_col]
                            iter_rows = max_in_col_ar[~max_non_match_row]
                            
                            for i in iter_rows: # iterate over matched/determined rows and actually add them to the output
                                pos = temp_pos[i]
                                #print(max_in_row, i, classes)
                                name = classes[max_in_row[i]]

                                conf = confmat[0]
                                if r!=1:
                                    conf = confmat[i,max_in_row[i]]
                                output.append({"name": str(name), "x": pos["x"], "y": pos["y"], "conf": float(conf)})

                                #print(classes[max_in_row_ar[i]], confmat[i, :], max_in_row_ar[i])

                            temp_pos = temp_pos[max_non_match_row]
                            classes = classes[max_non_match_col]
                            if len(temp_pos) == 0:
                                break

                            print(f"Reaching another iteration as the class ball(s) on {temp_pos} are not the highest confidence in their top1 classes. Now trying for {classes}")
                            confmat = confmat[max_non_match_row, max_non_match_col] # update confmat to new dimensions
                            r = confmat.shape[0] # check if there are any remaining rows

        if self.debug: print(f"Detected objects: {output}")
        print(f"Elapsed time for BallDetector.detect: {timer()-startTime}")
        return {"results": output, "mode": self.mode}


    def toRealDim(self, results, dimensionsTable):#(rw,rh)):
        """Returns results["results"] but with x and y as floats 
        """
        rw, rh = tuple(dimensionsTable)
        trans = []
        for r in results["results"]:
            rx, ry = r["x"]/self.w*rw, r["y"]/self.h*rh
            trans.append({"name": r["name"], "x": rx, "y": ry, "conf": float(r["conf"])})

        return trans

    def verify(self, img, results):
        """Overlay an image with its results and save it
        
        :param image: cv2 image on which the results where inferred using BallDetector.detect
        :param results: output of BallDetector.detect
        """
        output = results["results"]
        for r in output:
            x,y = r["x"],r["y"]
            color = (0,0,0)
            cv2.drawMarker(img, (x,y), color=color, markerType=cv2.MARKER_CROSS)
            cv2.putText(img, f"{r['name']}: {r['conf']:.2f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

        cv2.imwrite("verifyBallDetection.png", img)



if __name__=="__main__":
    #img = cv2.imread("images/image-73.png")
    img = "../image-90.png"
    if os.path.exists(img):
        img = cv2.imread(img)
    else:
        print("File not found.")
        exit
    b = BallDetector(debug=False, mode="8pool-detail")
    out = b.detect(img)
    b.verify(img, out)
    #micrarmat(np.array([3,1,2,1,0]),4,5)
    #micrarmat(np.array([3,1,2,0]),4,5)