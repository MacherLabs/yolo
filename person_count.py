#!/usr/bin/env python

import common
import cv2
from multitracker import KalmanTracker
from yolo import ObjectDetectorYolo

url = 'rtsp://admin:admin@192.168.0.10/live'
url = "/home/aestaq/Videos/persons.mp4"
# url = 1

LINE = {'x1': 0, 'y1': 160, 'x2': 600, 'y2': 160}
removalConfig = {
    'invisible_count': 35,
    'overlap_thresh': 0.9,
    'overlap_invisible_count': 2,
    'corner_percentage': 0.1,
    'corner_invisible_count': 5
    }


def get_pos(bbox):
    x, y, w, h = bbox
    cx, cy = x+w/2, y+h/2   # centroid coordinates
    side = 'Positive' if (cx-LINE['x1'])*(LINE['y2']-LINE['y1']) - (
        cy-LINE['y1'])*(LINE['x2']-LINE['x1']) >= 0 else 'Negative'
    return side


def draw_boxes(img, boxes, names):
    """ Draws bounding boxes of objects detected on given image """
    h, w = img.shape[:2]
    for box, tid in zip(boxes, names):
        # draw rectangle
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        text = "%d" % (tid)
        img = common.drawLabel(img, text, (x, y))
    return img


def demo_video(video_file):
    detector = ObjectDetectorYolo(model='tiny-yolo-voc')
    mtracker = KalmanTracker(['person'], tracker='deep_sort')

    cap = common.VideoStream(video_file, queueSize=4).start()
    cv2.waitKey(500)
    Outcount, Incount = 0, 0
    total_t, counter = 0, 0

    while not cap.stopped:
        t = common.clock()
        imgcv = cap.read()

        if imgcv is not None:
            counter += 1
            detections = detector.run(imgcv)
            mtracker.update(imgcv, detections)
            cvboxes, ids = [], []

            for tid, tracker in mtracker.trackers.iteritems():
                if tracker.consecutive_invisible_count < 5:
                    state_current = get_pos(tracker.bbox)

                    try:
                        if state_current != tracker.regionside:
                            tracker.statechange += 1
                            print state_current, tracker.regionside, tracker.statechange
                            if state_current == 'Positive':
                                if tracker.statechange % 2:
                                    Incount += 1
                                else:
                                    Outcount -= 1
                            else:
                                if tracker.statechange % 2:
                                    Outcount += 1
                                else:
                                    Incount -= 1
                            tracker.regionside = state_current

                    except AttributeError:
                        tracker.regionside = state_current
                        tracker.statechange = 0

                    cvboxes.append(tracker.bbox)
                    ids.append(tid)
            print Incount, Outcount

            cv2.line(imgcv, (LINE['x1'], LINE['y1']), (LINE['x2'], LINE['y2']),
                     (0, 0, 255), 4)
            common.drawLabel(imgcv, "IN:%d  OUT:%d" % (Incount, Outcount),
                             (10, 10), size=1, color=(0, 0, 255))
            common.showImage(draw_boxes(imgcv, cvboxes, ids))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        t1 = common.clock()
        dt = t1-t
        t = t1
        total_t += dt
        print counter/total_t


if __name__ == '__main__':
    demo_video(url)
