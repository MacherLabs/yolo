#!/usr/bin/env python
import os

#WORK_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = '/LFS'
MODEL_DIR = 'weights'
CFG_DIR = 'cfg'


class ObjectDetectorYolo(object):
    """
    A general obect detection class based on yolo darknet model.
        Required:
            model(str): yolo model name (default:'yolo')
            threshold(float): threshold of the given model (default:0.4)
            gpu(float): percentage of gpu use desired (default:0 running entirely on cpu)
    """
    def __init__(self, model='yolo', threshold=0.4, gpu=0.0):
        from darkflow.net.build import TFNet
        options = {
                    'model': os.path.join(WORK_DIR, CFG_DIR, model + '.cfg'),
                    'config': os.path.join(WORK_DIR, CFG_DIR),
                    'load': os.path.join(WORK_DIR, MODEL_DIR, model + '.weights'),
                    'threshold': threshold,
                    'gpu': gpu
                    }
        self.tfnet = TFNet(options)

    # Run detector on given frame and return list of objects
    def run(self, imgcv):
        result = self.tfnet.return_predict(imgcv)
        return self._format_result(result)

    # Format the results
    def _format_result(self, result):
        out_list = []
        for res in result:
            formatted_res = dict()
            formatted_res["class"] = res["label"]
            formatted_res["prob"] = float(res["confidence"])
            formatted_res["box"] = {"topleft": res["topleft"],
                                    "bottomright": res["bottomright"]}
            out_list.append(formatted_res)
        return out_list


if __name__ == '__main__':
    print(ObjectDetectorYolo.__doc__)
    import sys
    import pprint
    import cv2

    detector = ObjectDetectorYolo(model='yolo')
    image_url = '../test.jpg' if len(sys.argv) < 2 else sys.argv[1]
    imgcv = cv2.imread(image_url)
    if imgcv is not None:
        results = detector.run(imgcv)
        pprint.pprint(results)
    else:
        print("Could not read image: {}".format(image_url))
