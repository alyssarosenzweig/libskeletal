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
    return x if (x > 0 and x < 256) else (0 if x < 0 else 255)

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
    rgb   = cv2.resize(cv2.imread(prefix + str(number) + "_rgb.png"),   (0, 0), fx=0.25, fy=0.25)
    parts = cv2.resize(cv2.imread(prefix + str(number) + "_parts.png"), (0, 0), fx=0.25, fy=0.25)
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), skin(rgb), foreground(rgb), cv2.cvtColor(parts, cv2.COLOR_BGR2GRAY))

def randoffset(sd):
    return np.array([int(random.uniform(-sd, sd)), int(random.uniform(-sd, sd))])

def randvec(_):
    if random.random() > 0.5:
        return (randoffset(100), randoffset(100))
    else:
        return (randoffset(100), np.array([0, 0]))

def randpoint(img):
    return np.array([int(random.uniform(0, 256)), int(random.uniform(0, 256))])

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

def ugrabA(offset):
    return 256 if offset < 0 else 256 - offset

def lgrabA(offset):
    return 0 if offset > 0 else -offset

def ugrabB(offset):
    return 256 if offset > 0 else 256 + offset

def lgrabB(offset):
    return offset if offset > 0 else 0

def select(w, h, mat, offset, C):
    out = np.ones((w, h)) * C
    out[lgrabA(offset[0]):ugrabA(offset[0]), lgrabA(offset[1]):ugrabA(offset[1])] = mat[lgrabB(offset[0]):ugrabB(offset[0]), lgrabB(offset[1]):ugrabB(offset[1])]
    return out

def visualize(model):
    (clf, offsets) = model

    img = process(0)
    vis = img[0].copy()
    samples = np.zeros((256, 256, 15))
    gamma = gammamat(img)

    print "Sampling..."
    for f in xrange(0, 15):
        (u, v) = offsets[f]
        #print offsets[f]
        #U = gamma[0 + u[0] : 256 + u[0], 0 + u[1] : 256 + u[1]]
        #V = gamma[0 + v[0] : 256 + v[0], 0 + v[1] : 256 + u[1]]
        #U = gamma[lgrab(u[0]) : ugrab(u[0]), lgrab(u[1]) : ugrab(u[1])]
        #V = gamma[lgrab(v[0]) : ugrab(v[0]), lgrab(v[1]) : ugrab(v[1])]
        U = select(256, 256, gamma, u, 255)
        V = select(256, 256, gamma, v, 255)

        #print U
        #print V

        samples[:, :, f] = U - V
        #samples[x*256 + y] = sample(gamma, np.array([x, y]), offsets)

    print "Predicting..."
    vis = clf.predict(samples.reshape(256*256, 15)).reshape(256,256)
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
visualization = visualize(model)

print("Visualizing...")
cv2.imshow("Visualization", visualization)
cv2.waitKey(0)
