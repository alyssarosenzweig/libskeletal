import cv2
import numpy as np

rgb = cv2.imread("render_0_rgb.png")
parts = cv2.imread("render_0_parts.png")

cv2.imshow("parts", parts)

def skin(rgb):
    channels = cv2.split(rgb)
    out = (channels[2] * 0.6 - channels[1] * 0.3 - channels[0] * 0.3) - 10
    return out

def foreground(rgb):
    channels = cv2.split(rgb)
    (r, d) = cv2.threshold(channels[0] - 0x8C, 0, 255, cv2.THRESH_BINARY)
    return d

def findPart(parts, color):
    delta = np.array([4, 4, 4])
    return cv2.inRange(parts, color - delta, color + delta)

cv2.imshow("skin", skin(rgb))
cv2.imshow("fg", foreground(rgb))
cv2.imshow("head", findPart(parts, np.array([0x00, 0x00, 0x5B])))

def delta(mats, weights, pt, u, v):
    print(pt + u)
    print(pt + v)

delta(None, None, np.array([100, 100]), np.array([50, 50]), np.array([-50, 50]))

cv2.waitKey(0)
