# README 

A general obect detection class based on yolo darknet model.

### Inputs ###

* model(str): yolo model name (default:'yolo')
* threshold(float): threshold of the given model (default:0.4)
* gpu(float): percentage of gpu use desired (default:0 running entirely on cpu) 

### Requirements ###

* tensorflow
* cython
* darkflow
* git-lfs

### Installation ###
```sh
pip install --user cython
pip install --user git+https://github.com/MacherLabs/darkflow.git
```
```sh
pip install --user git+<https-url>
```
or
```sh
pip install --user git+ssh://git@bitbucket.org/macherlabs/yolo.git
```

### Usage ###

    from yolo import ObjectDetectorYolo
    detector = ObjectDetectorYolo(model='yolo')
    image_url = '../test.jpg'
    imgcv = cv2.imread(image_url)
    if imgcv is not None:
        results = detector.run(imgcv)
        pprint.pprint(results)
    else:
        print("Could not read image: {}".format(image_url))# yolo
