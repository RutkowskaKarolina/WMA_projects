import cv2 as cv
import pyautogui
from collections import deque
import numpy as np


# Extract meaningful ORB features from an image
def prepare(orb, img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def resize(frame):
    height, width, _ = frame.shape
    screen_height, screen_width = pyautogui.size()
    scale = screen_width / width
    new_height = int(height * scale)
    return cv.resize(frame, (screen_width, new_height))


# Initialize ORB with more features
orb = cv.ORB_create(1000)

images = [cv.imread(f'materials/saw{i}.jpg') for i in range(1, 5)]
ref_kp_des = [prepare(orb, img) for img in images]

cap = cv.VideoCapture('materials/sawmovie.mp4')

# Brute-force matcher with Hamming distance (for binary descriptors)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Buffer for bounding box smoothing (average over last 5 frames)
bbox_buffer = deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    best_matches = []
    best_kp = kp_frame

    # Compare image to the current frame
    for ref_kp, ref_des in ref_kp_des:
        if des_frame is None or ref_des is None:
            continue
        matches = bf.match(ref_des, des_frame)
        good = [m for m in matches if m.distance < 60]
        if len(good) > len(best_matches):
            best_matches = good  # Keep the strongest match

    # Do only if the object is confidently detected
    if len(best_matches) > 30:
        pts = [best_kp[m.trainIdx].pt for m in best_matches]  # Coordinates of matched points
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]

        # Compute center of mass of the matches
        cx = sum(xs) // len(xs)
        cy = sum(ys) // len(ys)

        # Filter out outlier points too far from the cluster center
        xs = list(filter(lambda x: abs(x - cx) < 150, xs))
        ys = list(filter(lambda y: abs(y - cy) < 150, ys))

        if xs and ys:
            # Dynamically calculate margin based on object size
            padding = max((max(xs) - min(xs)) // 6, 30)
            x1, y1 = min(xs) - padding, min(ys) - padding
            x2, y2 = max(xs) + padding, max(ys) + padding

            # Add to smoothing buffer and compute average box
            bbox_buffer.append((x1, y1, x2, y2))
            avg_box = np.mean(bbox_buffer, axis=0).astype(int)
            x1, y1, x2, y2 = avg_box

            frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('Chainsaw movie', frame)

    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
