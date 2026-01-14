# Trained YOLO models

The output from a training process is a `.pt` file. We copy this into this directory and export it into TensorRT's `.engine` format using the [official ultralytics guide](https://docs.ultralytics.com/de/integrations/tensorrt/#usage).
- Command for a detail model: `yolo export model=detailModel-11-m.pt format=engine imgsz=160 batch=24`
    - As we have a maximum of 16 billard balls on the field at the same time, we don't need a high batch number. For safety (a higher batch can cause errors), a batch size of 24 allows for a lot of false positive detections.
    - We also use two models with `batch=24` and `batch=4`, as less images passed as a batch than 4 result in a crash in the larger batch model.