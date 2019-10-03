import numpy as np, cv2
import matplotlib.pyplot as plt
import warnings, msvcrt
import vidutils # I am haveing trouble using cv2 to write videos so I made a module to read and write frame data
import os, sys, time
from numba import njit
from multiprocessing import Process, Pipe, Queue, Event

# TODO log errors (and attempt recovery)

# Process schematic
# Main, collect the frames and send them to the necessary places
# Process, scans

#### Program Goals
# figure out how to record if human is detected
# only record frames in between the checks so I only need to keep a matrix of all the frams since last check in memory
# if human is detected, get a still of the face
#### =======

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) # CAP_DSHOW allows change in resolution
if not cap.isOpened():
    # log error
    sys.exit()


# get resolution of camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)

h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # height        # in case resulution has not been set
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
c = 3                                       # three color channels
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30

## Shared resource management
conn_in, conn_out = Pipe(duplex=False)  # to send frames to be scanned
bufq = Queue()                          # to send array buffers to be written
okay_to_send = Event()                  # to denote when it is okay to send
detection_start = Event()               # to trigger when detection occurs
detection_in = Event()
detection_end = Event()                 # to trigger when detection event ends
quit_condition = Event()                # to trigger when q is pressed
buf_request = Event()                   # to trigger buffer transmission

## Global contants
frame = np.zeros((h,w,c))               # memory is preallocated for the main process
running = True                          # condition for program running


# calculation the magnitude of an array as the sum of the norms of each pixel
# I can try other norms later to test how it affects the motion detection
@njit
def mag(arr):
    h = arr.shape[0]
    w = arr.shape[1]
    c = arr.shape[2]
    tot = 0
    for j in range(h):
        for i in range(w):
            tot += np.sqrt((arr[j,i]**2).sum())
    return tot


# pipes and queues are too slow, can try an inline program or code in c++
def scan_frame_motion(conn,okay_to_send,detection_start,detection_end,detection_in):
    # takes difference between current frame and last frame and preforms morphological opening
    global h, w, c, running
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # for dilation and erosion

    cframe0 = np.zeros((h,w,c)) # intialize frame
    cframe0 = conn.recv()       # get initial frame for initial subtraction
    okay_to_send.set()

    t0 = time.time()
    min_record = 10             # minimum time to record in seconds to prevent flickering
    motion_score_thresh = 6e10  # arbitrary number found from experiments

    cframe = np.zeros((h,w,c))
    while running:
        cframe = conn.recv()
        if not running:
            break
        dframe = cframe-cframe0
        cframe0 = cframe.copy()

        frame0 = frame
        dframe = cv2.erode(dframe, kernel, iterations=10)
        dframe = cv2.dilate(dframe, kernel, iterations=10)

        motion_score = mag(dframe)
        print(f'current motion score {np.log10(motion_score)}')

        # if there is a detection set the detection flag sequence
        if (motion_score>motion_score_thresh)&(not detection_in.is_set()):
            print(f'setting detection: ms {motion_score}')
            t0 = time.time()
            detection_in.set()
            detection_start.set()
        if (motion_score>motion_score_thresh)&(not detection_in.is_set()):
            t0 = time.time()
        if (motion_score<motion_score_thresh)&(detection_in.is_set()):
            if (time.time()>(t0+min_record)):
                print(f'clearing detection: ms {motion_score} {time.time()-t0}')
                detection_in.clear()
                detection_end.set()

        okay_to_send.set()


# main loop, gets frames and sends them to other processes


if __name__=='__main__':
    if cap.isOpened():

        p_scan = Process(target=scan_frame_motion,args=(conn_in,okay_to_send,detection_start,detection_end,detection_in))
        p_scan.start()

        # initialize events
        okay_to_send.set()
        detection_in.clear()
        detection_start.clear()
        detection_end.clear()
        i = 0 # increment as long as i < buf_size
        while cap.isOpened():

            ret, frame = cap.read()

            # send to scanner
            if okay_to_send.is_set():
                okay_to_send.clear()
                conn_out.send(frame) # send frame when okay
            if detection_start.is_set():
                # create writer when start signal is set
                fname = f".\\detections\\out{time.strftime('%a%d%b%I_%M_%S%p')}.jmov"
                writer = vidutils.VideoWriter(fname,(h,w,c),30)
                detection_start.clear() # clear so writer is not created twice
            if detection_in.is_set():
                writer.write(frame) # write frames while in a detection
            if detection_end.is_set():
                # close writer when close event is set
                writer.close()
                detection_end.clear() # don't close the same writer twice


            if not ret:
                running = False

                # log error
                # attempt to restart the program
                break

            # check if quit condition is set




    running = False
    p_scan.join() # program is catching here, need to use ctrl-c to stop

    cap.release()
    cv2.destroyAllWindows()
