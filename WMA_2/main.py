import cv2
import numpy as np
import os

folder_path = 'resources'
for i in range(1, 9):
    filename = os.path.join(folder_path, f'tray{i}.jpg')
    print(f"\n=== Analiza pliku: tray{i}.jpg ===")

    img = cv2.imread(filename)  # Wczytanie obrazu
    if img is None:
        print(f"Nie udało się wczytać {filename}")
        continue

    img_blur = cv2.medianBlur(img, 3)  # Rozmycie obrazu (usuwanie szumów)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)  # Konwersja do szarości
    edges = cv2.Canny(img_gray, 500, 650, apertureSize=5)  # Wykrywanie krawędzi (Canny)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=50, maxLineGap=5)  # Wykrywanie linii prostych
    if lines is None:
        print("Nie wykryto linii.")
        continue

    # Wyznaczenie prostokąta obejmującego tacę
    x_coords = []
    y_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    low_x = min(x_coords)
    high_x = max(x_coords)
    low_y = min(y_coords)
    high_y = max(y_coords)

    cv2.rectangle(img, (low_x, low_y), (high_x, high_y), (255, 255, 255), 3)  # Rysowanie prostokąta wokół tacy

    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=100, param2=35, minRadius=20, maxRadius=40)  # Wykrywanie monet (okręgów)

    count_inside = {'big': 0, 'small': 0}
    count_outside = {'big': 0, 'small': 0}

    if circles is not None:
        circles = np.uint16(np.around(circles))  # Zaokrąglenie współrzędnych monet
        for circle in circles[0]:
            x, y, r = circle
            if low_x < x < high_x and low_y < y < high_y:
                if r > 31:
                    count_inside['big'] += 1  # Duża moneta (5 zł) na tacy
                    color = (100, 0, 255)
                else:
                    count_inside['small'] += 1  # Mała moneta (5 gr) na tacy
                    color = (255, 0, 0)
                cv2.circle(img, (x, y), 3, (0, 0, 0), 4)  # Punkt na środku monety
            else:
                if r > 31:
                    count_outside['big'] += 1  # Duża moneta poza tacą
                    color = (100, 0, 255)
                else:
                    count_outside['small'] += 1  # Mała moneta poza tacą
                    color = (255, 0, 0)
            cv2.circle(img, (x, y), r, color, 2)  # Rysowanie okręgu wokół monety

    total_inside = count_inside['big'] * 5 + count_inside['small'] * 0.05  # Suma monet na tacy
    total_outside = count_outside['big'] * 5 + count_outside['small'] * 0.05  # Suma monet poza tacą
    total = round(total_inside + total_outside, 2)  # Całkowita suma

    print('5 zł wewnątrz:', count_inside['big'])
    print('5 zł na zewnątrz:', count_outside['big'])
    print('5 gr wewnątrz:', count_inside['small'])
    print('5 gr na zewnątrz:', count_outside['small'])
    print('Suma na tacy:', total_inside, 'zł')
    print('Suma poza tacą:', total_outside, 'zł')
    print('Łącznie:', total, 'zł')

    window_name = f'detected coins - tray{i}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(window_name, 450, 600)

    cv2.imshow(window_name, img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
