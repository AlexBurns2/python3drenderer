import cv2
import numpy as np

def generate_skybox(width=1280, height=720,
                    top_color=(255, 200, 150),
                    bottom_color=(255, 255, 255),
                    curvature=1.5):


    y = np.linspace(0, 1, height)[:, None]
    gradient = y ** curvature

    top_color = np.array(top_color, dtype=np.float32)
    bottom_color = np.array(bottom_color, dtype=np.float32)

    img = bottom_color + (top_color - bottom_color) * gradient
    img = np.clip(img, 0, 255).astype(np.uint8)

    img = np.tile(img, (1, width, 1))

    return img

if __name__ == "__main__":
    cv2.namedWindow("Procedural Skybox", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Procedural Skybox', 1920, 1080)
    skybox = generate_skybox(
        width=1280,
        height=720,
        top_color=(255, 180, 120),   
        bottom_color=(255, 255, 255), 
        curvature=2.2
    )

    cv2.imshow('Procedural Skybox', skybox)
    cv2.waitKey(0)
    cv2.destroyAllWindows()