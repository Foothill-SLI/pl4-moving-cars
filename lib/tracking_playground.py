import cv2
import numpy as np
import time
import os

dirname = os.path.dirname(__file__)
def relPath(path): return os.path.join(dirname, path)

def preprocess(img, shave=False, shavedim=(350,500, 500,1000)):
    #if the image is to be cropped make sure the values are sane first then reduce the image down to the new dimentions
    if shave:
        if(shavedim[0] < 0):
            shavedim[0] = 0
        if (shavedim[1] > img.shape[0]):
            shavedim[1] = img.shape[0]
        if (shavedim[2] < 0):
            shavedim[2] = 0
        if (shavedim[3] > img.shape[1]):
            shavedim[3] = img.shape[1]
        img = img[shavedim[0]:shavedim[1],shavedim[2]:shavedim[3]]
    sizexy = [img.shape[1], img.shape[0]]

    #get the appropriate padding on the image to make it square
    padhw = [0,0]
    if(sizexy[0] > sizexy[1]):
        dif = sizexy[0] - sizexy[1]
        border = cv2.copyMakeBorder(img, int(dif/2), int(dif/2), 0, 0, cv2.BORDER_CONSTANT, value=[200, 200, 200])
        padhw[0] = int(((dif/2)/border.shape[0]) * 448)

    elif (sizexy[1] > sizexy[0]):
        dif = sizexy[1] - sizexy[0]
        border = cv2.copyMakeBorder(img, 0, 0, int(dif / 2), int(dif / 2), cv2.BORDER_CONSTANT, value=[200, 200, 200])
        padhw[1] = int(((dif / 2) / border.shape[1]) * 448)
    else:
        border = img

    #resize the image to fit the 448,448 input that yolo requires
    border = cv2.resize(border, (448, 448))

    #yolo requires the image to be fed in by (channel, y,x). Transpose to match that.
    transposed = np.transpose(border, [2, 0, 1])
    return transposed, padhw, shavedim, border

(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(mosse.upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

vs = cv2.VideoCapture(relPath("../videos/cars.mp4"))

while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    grabed, frame = vs.read()

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    processed, padhw, shavedim, resized = preprocess(frame)
    frame = resized
    ##frame = cv2.resize(frame, (int(frame.shape[1]/3), int(frame.shape[0]/3)))
    (H, W) = frame.shape[:2]

    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a   bounding
    # box to track
    if key == ord("s"):
        tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
