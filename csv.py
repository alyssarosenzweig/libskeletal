import cv2
import numpy as np

prefix = "/home/alyssa/synthposes/dataset/render_"

def skin(rgb):
    channels = cv2.split(rgb)
    out = (channels[2] * 0.6 - channels[1] * 0.3 - channels[0] * 0.3) - 10
    return (out > 0)*0

def foreground(rgb):
    channels = cv2.split(rgb)
    (r, d) = cv2.threshold(channels[0] - 0x8C, 0, 255, cv2.THRESH_BINARY)
    return d

def findPart(parts, color):
    delta = np.array([4, 4, 4])
    return cv2.inRange(parts, color - delta, color + delta)

def index(mat, pt):
    apt = pt

    if pt[0] < 0:
        apt[0] = 0
    elif pt[0] > np.size(mat, 1):
        apt[0] = mat[pt]
        
    if pt[1] < 0:
        apt[1] = 0
    elif pt[1] > np.size(mat, 0):
        apt[1] = mat[pt]

    return mat[apt[0], apt[1]]

def delta(mats, weights, pt, u, v):
    (rgb, skin, foreground) = mats
    (left, right) = (pt + u, pt + v)

    colorScore = index(rgb, left)**2        - index(rgb, right)**2
    skinScore  = index(skin, left)**2       - index(skin, right)**2
    fgScore    = index(foreground, left)**2 - index(foreground, right)**2

    return np.dot(np.array([colorScore, skinScore, fgScore]), weights)

def process(number):
    rgb = cv2.imread(prefix + str(number) + "_rgb.png")
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), skin(rgb), foreground(rgb))


d = delta(process(0),
        np.array([1, 10, 100]),
        np.array([500, 500]),
        np.array([-8, 0]), np.array([0, 0]))

print(d)

# cv2.imshow("parts", parts)
# cv2.imshow("skin", skin(rgb))
# cv2.imshow("fg", foreground(rgb))
# cv2.imshow("head", findPart(parts, np.array([0x00, 0x00, 0x5B])))

# cv2.waitKey(0)
