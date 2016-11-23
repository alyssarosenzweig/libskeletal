import random, math
import cv2
import numpy as np

from sklearn.ensemble import RandomForestClassifier 

prefix = "/home/alyssa/synthposes/dataset/render_"

def skin(rgb):
    chans = cv2.split(rgb)
    return 1 * (((chans[2] * 0.6 - chans[1] * 0.3 - chans[0] * 0.3) - 10) > 0)

def foreground(rgb):
    return cv2.threshold(cv2.split(rgb)[0] - 0x8C, 0, 1, cv2.THRESH_BINARY)[1]

def cap(x):
    return x if (x > 0 and x < 1024) else (0 if x < 0 else 1023)

def index(mat, pt):
    try:
        return mat[pt[0], pt[1]]
    except IndexError:
        return 1000

def gammamat(mats):
    (gray, skin, foreground, _) = mats
    return np.float32(gray) + np.float32(foreground)*127 + np.float32(skin)*63

def delta(gamma, pt, offset):
    (u, v) = offset
    (left, right) = (pt + u, pt + v)

    return index(gamma, left) - index(gamma, right)

def process(number):
    rgb = cv2.imread(prefix + str(number) + "_rgb.png")
    parts = cv2.imread(prefix + str(number) + "_parts.png")
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), skin(rgb), foreground(rgb), parts)

def randoffset(sd):
    return np.array([int(random.uniform(-sd, sd)), int(random.uniform(-sd, sd))])

def randvec(_):
    if random.random() > 0.5:
        return (randoffset(100), randoffset(100))
    else:
        return (randoffset(100), np.array([0, 0]))

def randpoint(img):
    return np.array([int(random.uniform(0, 1024)), int(random.uniform(0, 1024))])

def sample(gamma, pt, offsets):
    return map(lambda off: delta(gamma, pt, off), offsets)

def generateFeatures(count):
    return map(randvec, [None] * count)

X = []
Y = []

def train(clf, features, no):
    img = process(no)
    gamma = gammamat(img)

    for i in range(1, 10000):
        pt = randpoint(img[0])

        Y.append(index(img[3], pt))
        X.append(sample(gamma, pt, features))

def visualize(model):
    (clf, offsets) = model

    img = process(49)
    vis = img[3].copy()
    gamma = gammamat(img)

    for x in range(50, 500):
        for y in range(200, 800):
            s = sample(gamma, np.array([x, y]), offsets)
            vis[x, y] = clf.predict(np.array(s).reshape(1, -1)) * 255
        cv2.imshow("Vis", vis)
        cv2.waitKey(1)

    return vis

print("Training...")

clf = RandomForestClassifier(n_estimators=1)

features = generateFeatures(15)
for image in range(0, 1):
    print "Image " + str(image)
    train(clf, features, image)

print "Fitting..."

clf = clf.fit(X, Y)
model = (clf, features)

print("Running...")
cv2.imshow("Visualization", visualize(model))
cv2.waitKey(0)
