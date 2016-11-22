import random, math
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

def isPart(parts, pt, color):
    c = index(parts, pt)
    delta = np.array([4, 4, 4])
    lower = color - delta
    upper = color + delta
    return 1 * (np.all(c > lower) and np.all(c < upper))

def index(mat, pt):
    apt = pt

    if pt[0] < 0:
        apt[0] = 0
    elif pt[0] >= np.size(mat, 1):
        apt[0] = np.size(mat, 1) - 1
        
    if pt[1] < 0:
        apt[1] = 0
    elif pt[1] >= np.size(mat, 1):
        apt[1] = np.size(mat, 1) - 1

    return mat[apt[0], apt[1]]

def delta(mats, weights, pt, offset):
    (u, v) = offset
    (rgb, skin, foreground, _) = mats
    (left, right) = (pt + u, pt + v)

    colorScore = index(rgb, left)**2        - index(rgb, right)**2
    skinScore  = index(skin, left)**2       - index(skin, right)**2
    fgScore    = index(foreground, left)**2 - index(foreground, right)**2

    return np.dot(np.array([colorScore, skinScore, fgScore]), weights)

def process(number):
    rgb = cv2.imread(prefix + str(number) + "_rgb.png")
    parts = cv2.imread(prefix + str(number) + "_parts.png")
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), skin(rgb), foreground(rgb), parts)

def randoffset():
    sd = 10
    return np.array([int(random.gauss(0, sd)), int(random.gauss(0, sd))])

def randvec(_):
    return (randoffset(), randoffset())

def randpoint(img):
    return np.array([int(random.uniform(0, np.size(img, 0))),
        int(random.uniform(0, np.size(img, 1)))])

DELTA_WEIGHTS = np.array([1, 10, 100])

def sample():
    features = 10
    print("isHead," + ",".join(map(lambda q: "d" + str(q), xrange(1, features))))
    offsets = map(randvec, [None] * features)

    img = process(0)

    for i in range(1, 10000):
        pt = randpoint(img[0])

        out = [isPart(img[3], pt, np.array([0x00, 0x00, 0x5B]))]

        for off in offsets:
            out.append(delta(img, DELTA_WEIGHTS, pt, off))

        print(",".join(map(str, out)))

#d = delta(process(0),
#        DELTA_WEIGHTS,
#        np.array([500, 500]),
#        randvec())

#print(d)

sample()

# cv2.imshow("parts", parts)
# cv2.imshow("skin", skin(rgb))
# cv2.imshow("fg", foreground(rgb))
# cv2.imshow("head", findPart(parts, np.array([0x00, 0x00, 0x5B])))

# cv2.waitKey(0)
