import numpy as np
import cv2


def show(img):
    cv2.imshow("bla", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    monalisa = cv2.imread(r"C:\Users\cohenass\PycharmProjects\PictureRecreation\data\monalisa_med.jpg")
    black = np.zeros((monalisa.shape[:2][0], monalisa.shape[:2][1], 3), np.uint8)
    c1 = np.zeros((monalisa.shape[:2][0], monalisa.shape[:2][1], 3), np.uint8)
    cv2.circle(c1, center=(100, 100), radius=80, color=(255, 0, 0), thickness=-1)
    #show(c1)

    c2 = np.zeros((monalisa.shape[:2][0], monalisa.shape[:2][1], 3), np.uint8)
    cv2.circle(c2, center=(100, 100), radius=50, color=(0, 255, 0), thickness=-1)
    #show(c2)

    c3 = np.zeros((monalisa.shape[:2][0], monalisa.shape[:2][1], 3), np.uint8)
    cv2.circle(c3, center=(300, 300), radius=80, color=(0, 0, 255), thickness=-1)
    #show(c3)

    merged = monalisa.copy()
    a = cv2.addWeighted(merged, 1, c1, 0.5, 0)
    # merged = cv2.addWeighted(merged, 1, c2, 0.5, 0)
    show(a)
    b = cv2.addWeighted(a, 1, c1, 0.5, 0)
    show(b)
    merged = cv2.addWeighted(merged, 1, c1, 0.5, 0)
    show(merged)
    merged = cv2.addWeighted(merged, 1, c1, 0.5, 0)

    show(merged)
