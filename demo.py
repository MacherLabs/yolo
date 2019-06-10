#!/usr/bin/env python

import cv2, os
import json
import argparse
from yolo import ObjectDetectorYolo
import pprint

cv2_version = cv2.__version__.split('.')[0]
SKIP_FRAMES = 4
VIDEO = 0     # 0:None, 1:Show, 2:Save, 3:Show and Save

def draw_bbox(img, detections):
    """ Draws bounding boxes of objects detected on given image """
    h, w = img.shape[:2]
    for obj in detections:
        # draw rectangle
        x1, y1, x2, y2 = obj['box']['topleft']['x'], obj['box']['topleft']['y'], obj['box']['bottomright']['x'], obj['box']['bottomright']['y']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
        # draw class text
        yoff = -10 if y1 > 20 else 20   # text remains inside image
        if cv2_version == '2':
            cv2.putText(img, obj['class'], (x1, y1+yoff), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.CV_AA)
        else:
            cv2.putText(img, obj['class'], (x1, y1+yoff), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return img

def process_video(video_url, detector, show):
    """ Run object detector on given video.
        Required:  
            video_url: str or int
            detector: an object of detector class
        Returns:
            out_list: a python list containing detected objects
    """
    results = []
    skip_count = 0      # number of frames skipped
    cap = cv2.VideoCapture(video_url)

    if show > 1:
        if cv2_version == '2':
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            width, height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fname = os.path.splitext(str(video_url))[0] + '_saved.avi'
        print 'Video written: ', fname
        videow = cv2.VideoWriter(fname, fourcc, fps, (width, height))

    while cap.isOpened():
        ret = cap.grab()
        if ret:
            skip_count += 1

            if skip_count > SKIP_FRAMES:
                skip_count = 0
                ret, imgcv = cap.retrieve()
                if not ret:
                    print "Cannot read video frame"
                    continue

                if cv2_version == '2':
                    video_msec = int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))    # get current video frame's position in time 
                else:
                    video_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))    # get current video frame's position in time 
                out_json = {"time":int(video_msec/1000), "objects":detector.run(imgcv)}
                print 'time:', video_msec/1000
                results.append(out_json)
                
                if show > 0:
                    img = draw_bbox(imgcv, out_json['objects'])
                if show%2 == 1:
                    if cv2_version == '2':
                        cv2.namedWindow(str(video_url), cv2.cv.CV_WINDOW_NORMAL)
                    else:
                        cv2.namedWindow(str(video_url), cv2.WINDOW_NORMAL)
                    cv2.imshow(str(video_url), img)
                if show > 1:
                    videow.write(img)
            
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break
        else:
            print "Cannot read video frame"
            break
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General object detection.')
    parser.add_argument('--video', type=str, help='video link or video file path')
    parser.add_argument('--image', type=str, help='image link or image file path')
    parser.add_argument('--camera', type=int, default=0, help='camera number (default:0)')
    parser.add_argument('--model', type=str, default='yolo', help='model name e.g. yolo, tiny-yolo')
    parser.add_argument('--gpu', type=float, default=0.0, help='gpu use e.g. 0.0 for cpu only and 1.0 for gpu only (default:0.0)')
    parser.add_argument('--show', type=int, default=0, choices=[0, 1, 2, 3], help='0:None, 1:ShowVideo, 2:SaveVideo, 3:ShowAndSaveVideo')
    args = parser.parse_args()

    print args.video, args.camera, args.image, args.model, args.show
    detector = ObjectDetectorYolo(model=args.model, gpu=args.gpu)
    results_json = []

    if args.video:
        results = process_video(args.video, detector, args.show)
        results = json.dumps(results)
        fw = open(os.path.splitext(str(args.video))[0] + '.json', 'w')
        fw.write(results)
        fw.close()
        cv2.destroyAllWindows()

    elif args.image:
        imgcv = cv2.imread(args.image)
        if imgcv is not None:
            out_json = {"time":0, "objects":detector.run(imgcv)}
            img = draw_bbox(imgcv, out_json['objects'])
            pprint.pprint(out_json)

            if cv2_version == '2':
                cv2.namedWindow(args.image, cv2.cv.CV_WINDOW_NORMAL)
            else:
                cv2.namedWindow(args.image, cv2.WINDOW_NORMAL)

            cv2.imshow(args.image, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    else:
        process_video(args.camera, detector, args.show)
        cv2.destroyAllWindows()
