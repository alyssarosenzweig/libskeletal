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
    channels = cv2.split(rgb)
    (r, d) = cv2.threshold(channels[0] - 0x8C, 0, 1, cv2.THRESH_BINARY)
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

def gammamat(mats):
    (gray, skin, foreground, _) = mats
    return gray*1 + skin*10 + foreground*100;

def delta(gamma, pt, offset):
    (u, v) = offset
    (left, right) = (pt + u, pt + v)

    return index(gamma, left) - index(gamma, right)

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

def sample(gamma, pt, offsets):
    out = []

    for off in offsets:
        out.append(delta(gamma, pt, off))

    return out

def train(features):
    offsets = map(randvec, [None] * features)

    img = process(0)
    gamma = gammamat(img)

    X = []
    Y = []

    for i in range(1, 50000):
        pt = randpoint(img[0])

        Y.append(isPart(img[3], pt, np.array([0x00, 0x00, 0x5B])))
        X.append(sample(gamma, pt, offsets))

    #clf = RandomForestClassifier(n_estimators=10)
    clf = linear_model.LogisticRegression()
    #clf = GaussianNB()
    clf = clf.fit(X, Y)

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
        cv2.waitKey(2)
        print x

    return vis

print("Training...")
model = train(100)
print("Running...")
cv2.imshow("Visualization", visualize(model))
cv2.waitKey(0)

#d = delta(process(0),
#        DELTA_WEIGHTS,
#        np.array([500, 500]),
#        randvec())

#print(d)



# cv2.imshow("parts", parts)
# cv2.imshow("skin", skin(rgb))
# cv2.imshow("fg", foreground(rgb))
# cv2.imshow("head", findPart(parts, np.array([0x00, 0x00, 0x5B])))

# cv2.waitKey(0)
