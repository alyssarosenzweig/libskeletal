import random, math
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, BernoulliNB

prefix = "/home/alyssa/synthposes/dataset/render_"

def skin(rgb):
    channels = cv2.split(rgb)
    out = (channels[2] * 0.6 - channels[1] * 0.3 - channels[0] * 0.3) - 10
    return (out > 0) * 1

def foreground(rgb):
    (r, d) = cv2.threshold(cv2.split(rgb)[0] - 0x8C, 0, 1, cv2.THRESH_BINARY)
    return d

def isPart(parts, pt, color):
    c = index(parts, pt)
    delta = np.array([4, 4, 4])
    return 1 * (np.all(c > color - delta) and np.all(c < color + delta))

def cap(x):
    return x if (x > 0 and x < 1024) else (0 if x < 0 else 1023)

def index(mat, pt):
    return mat[cap(pt[0]), cap(pt[1])]

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

def train(features):
    offsets = map(randvec, [None] * features)

    img = process(0)
    gamma = gammamat(img)
    #cv2.imshow("Gamma", gamma / 255)
    #return

    X = []
    Y = []

    for i in range(1, 9000):
        pt = randpoint(img[0])

        Y.append(isPart(img[3], pt, np.array([0x00, 0x00, 0x5B])))
        X.append(sample(gamma, pt, offsets))

    #clf = RandomForestClassifier(n_estimators=1).fit(X, Y)
    #clf = linear_model.LogisticRegression().fit(X, Y)
    clf = GaussianNB().fit(X, Y)

    return (clf, offsets)

def visualize(model):
    (clf, offsets) = model

    img = process(1)
    vis = img[0].copy()
    gamma = gammamat(img)

    for x in range(50, 1023):
        for y in range(400, 600):
            s = sample(gamma, np.array([x, y]), offsets)
            vis[x, y] = clf.predict(np.array(s).reshape(1, -1)) * 255
        cv2.imshow("Vis", vis)
        cv2.waitKey(1)

    return vis

print("Training...")
model = train(100)
print("Running...")
cv2.imshow("Visualization", visualize(model))
cv2.waitKey(0)
