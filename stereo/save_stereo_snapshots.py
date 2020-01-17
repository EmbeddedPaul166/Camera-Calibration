import cv2
import time
import sys
import os

def prepare_window():
    window_name = "Save snapshots"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowTitle(window_name, "Save snapshots")
    return window_name

def save_snapshots():

    name_left="left"
    name_right="right"
    folder_left="images/left/"
    folder_right="images/right/"
    
    video_capture_left = cv2.VideoCapture("v4l2src device=/dev/video3 ! video/x-raw,format=UYVY,width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink")
    video_capture_right = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw,format=UYVY,width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw,format=(string)BGR ! appsink")
    
    window_name = prepare_window()
    
    try:
        if not os.path.exists(folder_left):
            os.makedirs(folder_left)
            folder_left = os.path.dirname(folder_left)
            try:
                os.stat(folder_left)
            except:
                os.mkdir(folder_left)
                
        if not os.path.exists(folder_right):
            os.makedirs(folder_right)
            folder_right = os.path.dirname(folder_right)
            try:
                os.stat(folder_right)
            except:
                os.mkdir(folder_right)
    except:
        pass
    
    image_count   = 1
    w       = 1920
    h       = 1080

    file_name_left    = "%s/%s" %(folder_left, name_left)
    file_name_right    = "%s/%s" %(folder_right, name_right)
    while True:
        ret_left, frame_left = video_capture_left.read()
        ret_right, frame_right = video_capture_right.read()

        vertical_images = cv2.hconcat([frame_right, frame_left])
        cv2.imshow(window_name, vertical_images)


        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord(' '):
            print("Saving image ", image_count)
            cv2.imwrite("%s%d.jpg"%(file_name_left, image_count), frame_left)
            cv2.imwrite("%s%d.jpg"%(file_name_right, image_count), frame_right)
            image_count += 1

    video_capture_left.release()
    cv2.destroyAllWindows()

def main():
    save_snapshots()
    print("Files saved")

if __name__ == "__main__":
    main()



