import cv2
import numpy as np
import time
import os

debug = False

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker


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

def get_point(point):
    m_1= 0.6
    l_1= 140
    m_2= -0.35
    l_2= 390

    if point[1] >= m_2 * point[0] + l_2:
        return 0
    elif point[1] >= m_1 * point[0] + l_1:
        return 1
    else:
        return 2



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

dirname = os.path.dirname(__file__)
def relPath(path): return os.path.join(dirname, path)

net = cv2.dnn.readNetFromCaffe(relPath("../ssd/MobileNetSSD_deploy.prototxt.txt"),relPath("../ssd/MobileNetSSD_deploy.caffemodel"))
name = "GOPR1201"

vs = cv2.VideoCapture(relPath("../videos/" + name +".mov"))
multiTracker = cv2.MultiTracker_create()
writer = None
(w, h) = (None, None)

parking_lot = 0
faculty = 0

try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

multiTrackerList = []
current_centers = {}
paths = {}
counter = {}
i = 0
while True:
    i += 1
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    #frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if frame is None:
        print("sdfg")
        break

    processed, padhw, shavedim, resized = preprocess(frame)
    frame = resized

    #print(frame.shape)

    # get updated location of objects in subsequent frames
    toDelete = []
    for oneTracker in multiTrackerList:
        if counter[oneTracker] < 30:
            (success, box) = oneTracker.update(frame)

            # check to see if the tracking was a success
            if success:
                (x, y, W, H) = [int(v) for v in box]
                current_centers[oneTracker] = (x, y, x+W, y+H)
                if( len(paths[oneTracker]) >= 1 and (paths[oneTracker][-1][0] - (2 * x + W) / 2)**2 + (paths[oneTracker][-1][1] == (2 * y + H) / 2)**2 < 4 ):
                    counter[oneTracker] += 1
                else:
                    counter[oneTracker] = 0
                    paths[oneTracker].append( ((2 * x + W) / 2, (2 * y + H) / 2))
                cv2.rectangle(frame, (x, y), (x + W, y + H),
                    (0, 255, 0), 2)
            else:
                counter[oneTracker] = 100

    for eachCar, path in paths.items():
        if(counter[eachCar] < 10):
            for eachPoint in path[-20:]:
                frame = cv2.circle(frame, (int(eachPoint[0]), int(eachPoint[1])), 3, (255, 0, 0), 3)
            frame = cv2.circle(frame, (int(path[0][0]), int(path[0][1])), 3, (255,0,0),3)
        if(counter[eachCar] == 100 or counter[eachCar] == 30):
            start = get_point(path[0])
            end = get_point(path[-1])
            print(start)
            print(end)
            if start == 0 and end == 1:
                parking_lot += 1
            elif start == 0 and end == 2:
                faculty += 1
            elif start == 2 and end == 1:
                faculty -= 1
                parking_lot += 1
            elif start == 2 and end == 0:
                faculty -= 1
            elif start == 1 and end == 0:
                parking_lot -= 1
            counter[eachCar] = 1000
    # if the frame dimensions are empty, grab them
    if w is None or h is None:
        (w, h) = frame.shape[:2]

    if(i % 8 == 1):
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (448, 448), 127.5)
        net.setInput(blob)
        start = time.time()
        detections = net.forward()
        end = time.time()

        # loop over each of the layer outputs
        for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                if CLASSES[idx] == "car":
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (startX, startY, endX - startX, endY - startY))
                    toAdd = True
                    for tr, cen in current_centers.items():
                        cen_x_1 = (startX + endX) / 2
                        cen_y_1 = (startY + endY) / 2
                        cen_x_2 = (cen[0] + cen[2]) / 2
                        cen_y_2 = (cen[1] + cen[3]) / 2
                        if (cen_x_2  >= startX and cen_y_2 >= startY and cen_x_2 <= endX and cen_y_2 <= endY) or ((cen_x_1  >= cen[0] - 3 and cen_y_1 >= cen[1] - 3 and cen_x_1 <= cen[2] + 3 and cen_y_1 <= cen[3] + 3) or (cen_x_1 - cen_x_2)**2 + (cen_y_1 - cen_y_2)**2 < 400 ):
                            toAdd = False
                    if toAdd:
                        multiTrackerList.append(tracker)
                        current_centers[tracker] = (startX, startY, endX, endY)
                        paths[tracker] = []
                        counter[tracker] = 0
                    #cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    #y = startY - 15 if startY - 15 > 15 else startY + 15
                    #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)




    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if not debug:
            writer = cv2.VideoWriter(relPath("../test_counting/" + name + "-ssd_version3.avi"), fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if not debug:
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

    #cv2.line(frame, (25, 0), (25, 400), (0,255,0), 2)
    cv2.line(frame, (100,200), (200, 260), (0, 255, 0), 2)
    cv2.line(frame, (200, 320), (400, 250), (0,255,0), 2)
    cv2.putText(frame, "parking_lot - " + str(parking_lot) + " | faculty - " + str(faculty) , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    # write the output frame to disk
    if not debug:
        writer.write(frame)

if not debug:
    writer.release()
vs.release()
