import random, math
import cv2
import json
import numpy as np

from sklearn.ensemble  import RandomForestRegressor
from sklearn.cluster   import MeanShift
from sklearn.externals import joblib

prefix = "/home/alyssa/synthposes/private/render_"

SIZE = 64

ME = None
LIVE = False

# initialize background subtraction
if LIVE:
    stream = cv2.VideoCapture(0)
    bg = cv2.resize(stream.read()[1], (SIZE, SIZE))

def foreground(rgb):                                                            
    return cv2.threshold(cv2.split(rgb)[0] - 0x8C, 0, 1, cv2.THRESH_BINARY)[1]  

def skin(rgb):
    chans = cv2.split(rgb)
    return 1 * (((chans[2] * 0.6 - chans[1] * 0.3 - chans[0] * 0.3) - 8) > 0)

def gammamat(mats):
    (gray, skin, foreground, _) = mats
    return 0*np.float32(gray) + np.float32(foreground)*127 + np.float32(skin)*63

def process(number):
    global ME
    rgb   = cv2.resize(cv2.imread(prefix + str(number) + "_rgb.png"), (SIZE, SIZE))
    ME = rgb
    f = open(prefix + str(number) + "_skeleton.json")
    skel  = json.loads(f.read())
    f.close()
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), skin(rgb), foreground(rgb), skel)

def bgSubtract(rgb):
    return cv2.threshold(cv2.cvtColor(cv2.absdiff(rgb, bg), cv2.COLOR_BGR2GRAY), 32, 1, cv2.THRESH_BINARY)[1]

def process_stream():
    global ME
    rgb = cv2.resize(stream.read()[1], (SIZE, SIZE))
    ME = rgb
    fg = bgSubtract(rgb)
    return (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) & fg, skin(rgb), fg, "Cheater!")

def randoffset(sd):
    return np.array([int(random.uniform(-sd, sd)), int(random.uniform(-sd, sd))])

def randvec(_):
    if random.random() > 0.5:
        return (randoffset(SIZE / 4), randoffset(SIZE / 4))
    else:
        return (randoffset(SIZE / 4), np.array([0, 0]))

def generateFeatures(count):
    return map(randvec, [None] * count)

FEATURES = 100
COUNT    = 200
TRAINING = True

# internal joint order by the ML library
JOINTS = ["head", "lshoulder", "lelbow", "lhand", "rshoulder", "relbow", "rhand", "hip", "lpelvis", "lknee", "lfoot", "rpelvis", "rknee", "rfoot"]

def serialize_skeleton(skeleton):
    out = []
    
    for joint in JOINTS:
        out.append(skeleton[joint][0])
        out.append(skeleton[joint][1])

    return out

def unserialize_skeleton(skeleton):
    out = {}

    count = 0
    for joint in JOINTS:
        out[joint][0] = skeleton[count]
        out[joint][1] = skeleton[count + 1]

        count = count + 2

    return out

X = np.zeros((SIZE * SIZE * COUNT, FEATURES))
Y = np.zeros((SIZE * SIZE * COUNT, len(JOINTS) * 2), dtype=np.float32)

def distmapx(c):
    m = np.zeros((SIZE, SIZE), dtype=np.float32)
    for x in xrange(0, SIZE):
        m[:, x] = c - x
    return m.flatten()

def distmapy(c):
    m = np.zeros((SIZE, SIZE), dtype=np.float32)
    for y in xrange(0, SIZE):
        m[y, :] = c - y
    return m.flatten()

def train(clf, features, no):
    img = process(no)
    gamma = gammamat(img)

    for f in range(0, len(features)):
        (u, v) = features[f]

        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        X[no * SIZE*SIZE:(no+1) * SIZE*SIZE, f] = (U-V).reshape(SIZE*SIZE)

    skel = map(lambda x: int(x / (1024 / SIZE)), serialize_skeleton(img[3]))

    for k in xrange(0, len(JOINTS)):
        Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, k*2 + 0] = distmapx(skel[k*2])
        Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, k*2 + 1] = distmapy(SIZE - skel[k*2 + 1])

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

def predict(model, count):
    (clf, offsets) = model

    if LIVE:
        img = process_stream()
    else:
        img = process(COUNT + 1)

    vis = np.zeros((SIZE, SIZE), dtype=np.uint8)
    samples = np.zeros((SIZE, SIZE, count))
    gamma = gammamat(img)

    for f in xrange(0, count):
        (u, v) = offsets[f]
        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        samples[:, :, f] = U - V

    return clf.predict(samples.reshape(SIZE*SIZE, count))

def jointPos(vis, n):
    #X = vis[:, n*2 + 0] - distmapx(0)
    #Y = vis[:, n*2 + 1] - distmapy(0)
    #ms = MeanShift(bandwidth=4, bin_seeding=True, min_bin_freq=20)
    #ms.fit(np.column_stack([X, Y]))
    #print ms.cluster_centers_[0]
    #return tuple(map(int, ms.cluster_centers_[0]))
    I = np.reshape(vis[:, n*2]*vis[:, n*2] + vis[:, n*2+1]*vis[:, n*2+1], (SIZE, SIZE))
    (_1, _2, joint, _3) = cv2.minMaxLoc(cv2.GaussianBlur(I, (SIZE / 16 + 1, SIZE / 16 + 1), 0))
    return joint

if TRAINING:
    clf = RandomForestRegressor(n_estimators=1, n_jobs=4)

    features = generateFeatures(FEATURES)
    for image in range(0, COUNT):
        print "Image " + str(image)
        train(clf, features, image)

    clf = clf.fit(X, Y)
    model = (clf, features)

    joblib.dump(model, "model.pkl")
else:
    print "Loading model.."
    model = joblib.load("model.pkl")

visualization = predict(model, FEATURES)
visualization = np.abs(visualization)

r = SIZE / 64
c = (0, 255, 0)

while True:
    v = np.abs(predict(model, FEATURES))
    
    cv2.line(ME, jointPos(v, 0), jointPos(v, 1), c)
    cv2.line(ME, jointPos(v, 1), jointPos(v, 2), c)
    cv2.line(ME, jointPos(v, 2), jointPos(v, 3), c)
    cv2.line(ME, jointPos(v, 0), jointPos(v, 4), c)
    cv2.line(ME, jointPos(v, 4), jointPos(v, 5), c)
    cv2.line(ME, jointPos(v, 5), jointPos(v, 6), c)

    # left leg
    cv2.line(ME, jointPos(v, 4), jointPos(v, 7), c)
    cv2.line(ME, jointPos(v, 7), jointPos(v, 8), c)
    cv2.line(ME, jointPos(v, 8), jointPos(v, 9), c)
    cv2.line(ME, jointPos(v, 9), jointPos(v, 10), c)

    # right leg
    cv2.line(ME, jointPos(v, 1), jointPos(v, 11), c)
    cv2.line(ME, jointPos(v, 11), jointPos(v, 12), c)
    cv2.line(ME, jointPos(v, 12), jointPos(v, 13), c)

    cv2.circle(ME, jointPos(v, 0), r, (0, 0, 0), -1)

    cv2.circle(ME, jointPos(v, 1), r, (0, 255, 0), -1)
    cv2.circle(ME, jointPos(v, 2), r, (100, 255, 0), -1)
    cv2.circle(ME, jointPos(v, 3), r, (200, 255, 0), -1)

    cv2.circle(ME, jointPos(v, 4), r, (0, 0, 255), -1)
    cv2.circle(ME, jointPos(v, 5), r, (100, 0, 255), -1)
    cv2.circle(ME, jointPos(v, 6), r, (200, 0, 255), -1)
    
    cv2.circle(ME, jointPos(v, 7), r, (255, 255, 255), -1)
    
    cv2.circle(ME, jointPos(v, 7), r, (255, 0, 0), -1)
    cv2.circle(ME, jointPos(v, 8), r, (255, 127, 0), -1)
    cv2.circle(ME, jointPos(v, 9), r, (255, 255, 0), -1)
     
    cv2.circle(ME, jointPos(v, 10), r, (255, 0, 255), -1)
    cv2.circle(ME, jointPos(v, 11), r, (255, 127, 255), -1)
    cv2.circle(ME, jointPos(v, 12), r, (255, 255, 255), -1)

    cv2.imshow("me", cv2.resize(ME, (512, 512)))

    if cv2.waitKey(1) == 27:
        break
