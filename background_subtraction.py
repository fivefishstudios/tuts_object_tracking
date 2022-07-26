# background_subtraction.py
# motion tracking by background subtraction
# Opencv Computer Vision Projects with Python (Joseph Howse, Prateek Joshi, Michael Beyeler) (z-lib.org).pdf, page 270
# 7/24/22

import cv2

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('../_video/RoadTraffic.mp4')

    # create bkgd subtractor object
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
    # bgSubtractor = cv2.createBackgroundSubtractorKNN()

    # this controls the learning rate of the algorithm
    # learning rate about the background. Higher value for history
    # will mean slower learning rate
    history = 100

    # iterate until user presses ESC key
    while True:
        scaling_factor = 0.5
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_thresh = frame_gray.copy()
        # cv2.threshold(frame_gray, 160, 255, cv2.THRESH_BINARY, dst=frame_thresh)

        # apply the bkgd subtraction on model to the input frame
        mask = bgSubtractor.apply(frame_thresh, 0.5)

        # convert from grayscale to BGR
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow('input', frame)
        cv2.imshow('mask', mask & frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
