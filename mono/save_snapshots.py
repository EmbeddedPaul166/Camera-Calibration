import cv2
import time
import sys
import os


def prepare_window():
    window_name = "Save snapshots"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowTitle(window_name, "Save snapshots")
    return window_name

def save_snapshots():

    name="snapshot"
    folder="images/mono/"
    
    video_capture = cv2.VideoCapture("v4l2src device=/dev/video3 ! video/x-raw,format=UYVY,width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink")
    
    window_name = prepare_window()
    
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
            folder = os.path.dirname(folder)
            try:
                os.stat(folder)
            except:
                os.mkdir(folder)
    except:
        pass
    
    nSnap   = 1
    w       = 1920
    h       = 1080

    fileName    = "%s/%s_%d_%d_" %(folder, name, w, h)
    while True:
        ret, frame = video_capture.read()

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord(' '):
            print("Saving image ", nSnap)
            cv2.imwrite("%s%d.jpg"%(fileName, nSnap), frame)
            nSnap += 1

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    save_snapshots()
    print("Files saved")

if __name__ == "__main__":
    main()



