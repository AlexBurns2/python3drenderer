import cv2
import numpy as np

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print("mouse", x, y)

cv2.namedWindow("test")
cv2.setMouseCallback("test", on_mouse)

while True:
    img = 255 * np.ones((200,200,3), dtype=np.uint8)
    cv2.imshow("test", img)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cv2.destroyAllWindows()