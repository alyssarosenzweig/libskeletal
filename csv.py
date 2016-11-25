import random, math
import cv2
import numpy as np

from sklearn.ensemble import RandomForestClassifier 

prefix = "/home/alyssa/synthposes/private/render_"

SIZE = 64

# initialize background subtraction
#stream = cv2.VideoCapture(0)
#bg = cv2.resize(stream.read()[1], (SIZE, SIZE))

def skin(rgb):
    chans = cv2.split(rgb)
    return 1 * (((chans[2] * 0.6 - chans[1] * 0.3 - chans[0] * 0.3) - 8) > 0)

def gammamat(mats):
    (gray, skin, foreground, _) = mats
    return 0*np.float32(gray) + np.float32(foreground)*127 + np.float32(skin)*63

def color_encode(mat):
    (blue, green, red) = cv2.split(mat)
    return 1*np.int32(blue) + 256*np.int32(green) + 65536*np.int32(red)

def color_decode(mat):
    blue  = np.float32( (mat & 0x0000FF) >>  0 ) / 256
    green = np.float32( (mat & 0x00FF00) >>  8 ) / 256
    red   = np.float32( (mat & 0xFF0000) >> 16 ) / 256

    return cv2.merge([blue, green, red])

def process(number):
    rgb   = cv2.resize(cv2.imread(prefix + str(number) + "_rgb.png"),   (SIZE, SIZE))
    parts = cv2.resize(cv2.imread(prefix + str(number) + "_parts.png"), (SIZE, SIZE))
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), skin(rgb), foreground(rgb), color_encode(parts))

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
        return (randoffset(12), randoffset(12))
    else:
        return (randoffset(12), np.array([0, 0]))

def generateFeatures(count):
    return map(randvec, [None] * count)

FEATURES = 100
COUNT = 98

X = np.zeros((4096 * COUNT, FEATURES))
Y = np.zeros((4096 * COUNT), dtype=np.uint32)

def train(clf, features, no):
    img = process(no)
    gamma = gammamat(img)

    for f in range(0, len(features)):
        (u, v) = features[f]

        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        X[no * SIZE*SIZE:(no+1) * SIZE*SIZE, f] = (U-V).reshape(SIZE*SIZE)

    Y[no*SIZE*SIZE:(no + 1)*SIZE*SIZE] = img[3].reshape(SIZE*SIZE)

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

    #img = process_stream()
    img = process(99)

    vis = np.zeros((SIZE, SIZE), dtype=np.uint8)
    samples = np.zeros((SIZE, SIZE, count))
    gamma = gammamat(img)

    cv2.imshow("Gamma", cv2.resize(gamma / 256, (512, 512)))

    for f in xrange(0, count):
        (u, v) = offsets[f]
        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        samples[:, :, f] = U - V

    vis = clf.predict(samples.reshape(SIZE*SIZE, count)).reshape(SIZE, SIZE)
    return color_decode(vis * img[2])

clf = RandomForestClassifier(n_estimators=10)

features = generateFeatures(FEATURES)
for image in range(0, COUNT):
    print "Image " + str(image)
    train(clf, features, image)

clf = clf.fit(X, Y)
model = (clf, features)

visualization = cv2.resize(visualize(model, FEATURES), (512, 512))

cv2.imshow("Visualization", visualization)
cv2.waitKey(0)

#while True:
#    v = visualize(model, FEATURES)
#    cv2.imshow("Visualization", cv2.resize(v, (1024, 1024)))
#    if cv2.waitKey(1) == 27:
#        break
