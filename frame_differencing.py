# frame_differencing.py
# simplest technique to determine what parts of the video are moving
# 7/22/22

import cv2

def frame_diff(prev_frame, cur_frame, next_frame):
    # absolute difference between current frame and next frame
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # absolute difference between current frame and previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # return the result of bitwise 'AND' between the two resultant images
    return cv2.bitwise_and(diff_frames1, diff_frames2)


# capture the frame from webcam
def get_frame(cap):
    # capture the frame
    ret, frame = cap.read()

    # resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # return the grayscale image
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.5

    prev_frame = get_frame(cap)
    cur_frame = get_frame(cap)
    next_frame = get_frame(cap)

    # iterate until the user presses the ESC key
    while True:
        # display result of frame differencing
        frame_diff_result = frame_diff(prev_frame, cur_frame, next_frame)

        frame_diff_thresh = frame_diff_result.copy()
        cv2.threshold(frame_diff_result,20, 255, cv2.THRESH_BINARY_INV,dst=frame_diff_thresh)

        # find contours in the image
        contours = cv2.findContours(frame_diff_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours[0]:
            for contour in contours[0]:
                print('contour found!')
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(frame_diff_result, (x,y), (x+w, y+h), (0,255,0), 4)
            cv2.imshow('Object Movement', frame_diff_result)

        # update the variables
        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = get_frame(cap)

        # check if ESC key pressed
        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()


