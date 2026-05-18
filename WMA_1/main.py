import numpy as np
import cv2 as cv

cap = cv.VideoCapture('resources/movingball.mp4')
kernel = np.ones((5, 5), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv.resize(frame, (350, 500), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(resized_frame, cv.COLOR_BGR2HSV)

    # Maski koloru czerwonego
    lower_red1 = np.array([0, 100, 10])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 20])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Morfologia
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    # Znajdź kontury
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 100:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv.circle(resized_frame, center, radius, (0, 255, 0), 2)
            cv.circle(resized_frame, center, 5, (255, 0, 0), -1)
            break

    cv.imshow('mask', mask)
    cv.imshow('closing', closing)
    cv.imshow('tracked ball', resized_frame)

    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
