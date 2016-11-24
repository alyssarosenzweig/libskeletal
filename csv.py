import random, math
import cv2
import numpy as np

from sklearn.ensemble import RandomForestClassifier 

prefix = "/home/alyssa/synthposes/dataset/render_"

SIZE = 512

# initialize background subtraction
stream = cv2.VideoCapture(0)
bg = cv2.resize(stream.read()[1], (SIZE, SIZE))

def skin(rgb):
    chans = cv2.split(rgb)
    return 1 * (((chans[2] * 0.6 - chans[1] * 0.3 - chans[0] * 0.3) - 8) > 0)

def foreground(rgb):
    return cv2.threshold(cv2.split(rgb)[0] - 0x8C, 0, 1, cv2.THRESH_BINARY)[1]

def cap(x):
    return x if (x > 0 and x < SIZE) else (0 if x < 0 else SIZE - 1)

def index(mat, pt):
    try:
        return mat[pt[0], pt[1]]
    except IndexError:
        return 1000

def gammamat(mats):
    (gray, skin, foreground, _) = mats
    return 0*np.float32(gray) + np.float32(foreground)*127 + np.float32(skin)*63

def delta(gamma, pt, offset):
    (u, v) = offset
    (left, right) = (pt + u, pt + v)

    return index(gamma, left) - index(gamma, right)

def process(number):
    rgb   = cv2.resize(cv2.imread(prefix + str(number) + "_rgb.png"),   (SIZE, SIZE))
    parts = cv2.resize(cv2.imread(prefix + str(number) + "_parts.png"), (SIZE, SIZE))
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), skin(rgb), foreground(rgb), cv2.split(parts)[0])

def bgSubtract(rgb):
    return cv2.threshold(cv2.cvtColor(cv2.absdiff(rgb, bg), cv2.COLOR_BGR2GRAY), 32, 1, cv2.THRESH_BINARY)[1]

def process_stream():
    rgb = cv2.resize(stream.read()[1], (SIZE, SIZE))
    fg = bgSubtract(rgb)
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) & fg, skin(rgb), fg, "Cheater!")

def randoffset(sd):
    return np.array([int(random.uniform(-sd, sd)), int(random.uniform(-sd, sd))])

def randvec(_):
    if random.random() > 0.5:
        return (randoffset(100), randoffset(100))
    else:
        return (randoffset(100), np.array([0, 0]))

def randpoint(img):
    return np.array([int(random.uniform(0, SIZE)), int(random.uniform(0, SIZE))])

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
    return SIZE if offset < 0 else SIZE - offset

def lgrabA(offset):
    return 0 if offset > 0 else -offset

def ugrabB(offset):
    return SIZE if offset > 0 else SIZE + offset

def lgrabB(offset):
    return offset if offset > 0 else 0

def select(w, h, mat, offset, C):
    out = np.full((w, h), C)
    out[lgrabA(offset[0]):ugrabA(offset[0]), lgrabA(offset[1]):ugrabA(offset[1])] = mat[lgrabB(offset[0]):ugrabB(offset[0]), lgrabB(offset[1]):ugrabB(offset[1])]
    return out

def visualize(model, count):
    (clf, offsets) = model

    img = process_stream()

    vis = np.zeros((SIZE, SIZE), dtype=np.uint8)
    samples = np.zeros((SIZE, SIZE, count))
    gamma = gammamat(img)

    return gamma / 255
    print "Sampling..."
    for f in xrange(0, count):
        (u, v) = offsets[f]
        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        samples[:, :, f] = U - V

    print "Predicting..."
    vis = clf.predict(samples.reshape(SIZE*SIZE, count)).reshape(SIZE, SIZE)
    return vis * img[2]

print("Training...")

clf = RandomForestClassifier(n_estimators=1)

FEATURES = 30

features = generateFeatures(FEATURES)
for image in range(0, 10):
    print "Image " + str(image)
#    train(clf, features, image)

print "Fitting..."

#clf = clf.fit(X, Y)
model = (clf, features)

#print("Running...")
#visualization = visualize(model, FEATURES)

#print("Visualizing...")
#cv2.imshow("Visualization", visualization)
#cv2.waitKey(0)

print("Running...")

# show normal gamma for comp
cv2.imshow("Comp", gammamat(process(5)) / 255)

while True:
    cv2.imshow("Visualization", visualize(model, FEATURES))
    cv2.waitKey(1)
