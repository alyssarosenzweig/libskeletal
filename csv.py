import random, math
import cv2
import json
import numpy as np

from sklearn.ensemble import RandomForestRegressor

prefix = "/home/alyssa/synthposes/private/render_"

SIZE = 64

ME = None

# initialize background subtraction
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
    rgb   = cv2.resize(cv2.imread(prefix + str(number) + "_rgb.png"), (SIZE, SIZE))
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
        return (randoffset(12), randoffset(12))
    else:
        return (randoffset(12), np.array([0, 0]))

def generateFeatures(count):
    return map(randvec, [None] * count)

FEATURES = 100
COUNT = 20

# internal joint order by the ML library
#JOINTS = ["head", "lshoulder", "lelbow", "lhand", "rshoulder", "relbow", "rhand"]
JOINTS = ["head", "lhand", "rhand"]

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
    #return cv2.resize(m, (SIZE * SIZE, 1))

def distmapy(c):
    m = np.zeros((SIZE, SIZE), dtype=np.float32)
    for y in xrange(0, SIZE):
        m[y, :] = c - y
    #return cv2.resize(m, (SIZE*SIZE, 1))
    return m.flatten()

def train(clf, features, no):
    img = process(no)
    gamma = gammamat(img)

    for f in range(0, len(features)):
        (u, v) = features[f]

        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        X[no * SIZE*SIZE:(no+1) * SIZE*SIZE, f] = (U-V).reshape(SIZE*SIZE)

    print serialize_skeleton(img[3])

    (hx, hy, ix, iy, jx, jy) = map(lambda x: int(x / (1024 / SIZE)), serialize_skeleton(img[3]))
    hy = SIZE - hy # blender uses a flipped coordinate system
    iy = SIZE - iy # blender uses a flipped coordinate system
    jy = SIZE - jy # blender uses a flipped coordinate system
    Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, 0] = distmapx(hx)
    Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, 1] = distmapy(hy)
    Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, 2] = distmapx(ix)
    Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, 3] = distmapy(iy)
    Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, 4] = distmapx(jx)
    Y[no * SIZE*SIZE:(no + 1) * SIZE*SIZE, 5] = distmapy(jy)

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

    img = process_stream()
    #img = process(9)

    vis = np.zeros((SIZE, SIZE), dtype=np.uint8)
    samples = np.zeros((SIZE, SIZE, count))
    gamma = gammamat(img)

    #cv2.imshow("Gamma", cv2.resize(gamma / 256, (512, 512)))

    for f in xrange(0, count):
        (u, v) = offsets[f]
        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        samples[:, :, f] = U - V

    return clf.predict(samples.reshape(SIZE*SIZE, count))

def jointPos(vis, n):
    I = -np.reshape(vis[:, n*2] + vis[:, n*2+1], (SIZE, SIZE))
    m = cv2.moments(np.float32(I > -1))

    if m["m00"] == 0:
        return (-1, -1)

    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

clf = RandomForestRegressor(n_estimators=1)

features = generateFeatures(FEATURES)
for image in range(0, COUNT):
    print "Image " + str(image)
    train(clf, features, image)

clf = clf.fit(X, Y)
model = (clf, features)

visualization = predict(model, FEATURES)
visualization = np.abs(visualization)
print jointPos(visualization, 0)
#cv2.imshow("Y", np.reshape(visualization[:, 0] + visualization[:, 1], (SIZE, SIZE)) * 100)

#cv2.waitKey(0)

while True:
    v = np.abs(predict(model, FEATURES))
    cv2.circle(ME, jointPos(v, 0), 3, (255, 0, 0), -1)
    cv2.circle(ME, jointPos(v, 1), 3, (0, 255, 0), -1)
    cv2.circle(ME, jointPos(v, 2), 3, (0, 0, 255), -1)
    cv2.imshow("me", cv2.resize(ME, (512, 512)))
    #M = np.float32(np.reshape(v[:, 0] + v[:, 1], (SIZE, SIZE)))
    #cv2.imshow("M", M / 10)
    #cv2.imshow("Y", cv2.resize(M / 100, (512, 512)))
    #cv2.imshow("Visualization", cv2.resize(v, (512, 512)))
    if cv2.waitKey(1) == 27:
        break
