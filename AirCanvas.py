import cv2
import numpy as np
import HandTrackingModule as hTracker

cap = cv2.VideoCapture(0)
kernel = np.ones((10, 10), np.uint8)
PCP = 0
min_area = 1000
h1 = int(cap.get(4))
w1 = int(cap.get(3))

canvas = np.zeros((h1, w1, 3), np.uint8)

detector = hTracker.handDetector(DetectionConf=0.7)

while True:
    success, frame = cap.read()
    frame = detector.findHands(frame)
    x2, y2 = 0, 0
    colors = [(255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        x2, y2 = lmList[8][1], lmList[8][2]
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

    cv2.rectangle(frame, (20,1), (120,65), (122,122,122), -1)
    cv2.rectangle(frame, (140,1), (220,65), colors[0], -1)
    cv2.rectangle(frame, (240,1), (320,65), colors[1], -1)
    cv2.rectangle(frame, (340,1), (420,65), colors[2], -1)
    cv2.rectangle(frame, (440,1), (520,65), colors[3], -1)
    cv2.rectangle(frame, (540,1), (620,65), colors[4], -1)
    cv2.putText(frame, "CLEAR ALL", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (155, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "VIOLET", (255, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (355, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (465, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (555, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)

    lower_bound = np.array([000, 000, 000])
    upper_bound = np.array([255, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Get the biggest contour from all the detected contours
        cmax = max(contours, key=cv2.contourArea)

        # Find the area of the contour
        area = cv2.contourArea(cmax)
        M = cv2.moments(cmax)
        if area > min_area:

            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), 2)
            #print (x2, y2)

            if y2 < 65:
                    color = 0
                    # Clear all
                    if x2 > 20 and x2 < 139:
                        canvas = np.zeros((h1, w1, 3), np.uint8)

                    elif x2 > 140 and x2 < 219:
                        color = colors[0]

                    elif x2 > 220 and x2 < 339:
                        color = colors[1]

                    elif x2 > 340 and x2 < 439:
                        color = colors[2]

                    elif x2 > 440 and x2 < 539:
                        color = colors[3]

                    elif x2 > 540 and x2 < 620:
                        color = colors[4]
            if PCP != 0:
                cv2.line(canvas, PCP, (x2, y2), color, 2)
            PCP = (x2, y2)
        else:
            PCP = 0
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_binary = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
    canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
    #frame = cv2.bitwise_and(frame, canvas_binary)
    #frame = cv2.bitwise_or(frame, canvas)
    cv2.imshow("Frame", frame)
    #cv2.waitKey(1)
    cv2.imshow('Canvas', canvas)

    if cv2.waitKey(1) == ord("k"):
        break

cap.release()
cv2.destroyAllWindows()
