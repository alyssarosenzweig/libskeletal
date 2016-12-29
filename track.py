"""
libskeletal: human pose estimation from monocular video based on Shotton et al
Copyright (C) 2016 Alyssa Rosenzweig

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import random, math
import cv2
import json
import numpy as np

from sklearn.ensemble  import RandomForestRegressor
from sklearn.externals import joblib

prefix = "/home/alyssa/synthposes/private/render_"

# configuration metadata

FEATURES = 200
COUNT    = 500
LIVE     = True
TRAINING = False
SAVE     = True
SIZE     = 64

ME = None

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
    return 0*np.float32(gray) + np.float32(foreground)*63 + np.float32(skin)*np.float32(foreground)*127

def process(number, training):
    global ME
    rgb   = cv2.resize(cv2.imread(prefix + str(number) + "_rgb.png"), (SIZE, SIZE))
    ME = rgb
    
    if training:
        f = open(prefix + str(number) + "_skeleton.json")
        skel  = json.loads(f.read())
        f.close()
    else:
        skel = "Cheater!"

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
    img = process(no, True)
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
        img = process(COUNT + 3, False)

    vis = np.zeros((SIZE, SIZE), dtype=np.uint8)
    samples = np.zeros((SIZE, SIZE, count))
    gamma = gammamat(img)

    cv2.imshow("Gamma", gamma)

    for f in xrange(0, count):
        (u, v) = offsets[f]
        U = select(SIZE, SIZE, gamma, u, 255)
        V = select(SIZE, SIZE, gamma, v, 255)

        samples[:, :, f] = U - V

    return clf.predict(samples.reshape(SIZE*SIZE, count))

def jointPos(vis, n):
    I = np.reshape(vis[:, n*2]*vis[:, n*2] + vis[:, n*2+1]*vis[:, n*2+1], (SIZE, SIZE))
    return cv2.minMaxLoc(cv2.GaussianBlur(I, (SIZE / 4 + 1, SIZE / 4 + 1), 0))[2]

def trainModel(featureCount, imageCount, save):
    clf = RandomForestRegressor(n_estimators=1, n_jobs=-1)

    features = generateFeatures(featureCount)

    for image in range(0, imageCount):
        print "Image " + str(image)
        train(clf, features, image)

    clf = clf.fit(X, Y)
    model = (clf, features)

    if save:
        joblib.dump(model, "model.pkl")

    return model

def getModel():
    if TRAINING:
        return trainModel(FEATURES, COUNT, SAVE)j
    else:
        return joblib.load("model.pkl")

def visualizeSkeleton(img, v, r, c):
    cv2.line(img, jointPos(v, 0), jointPos(v, 1), c)
    cv2.line(img, jointPos(v, 1), jointPos(v, 2), c)
    cv2.line(img, jointPos(v, 2), jointPos(v, 3), c)
    cv2.line(img, jointPos(v, 0), jointPos(v, 4), c)
    cv2.line(img, jointPos(v, 4), jointPos(v, 5), c)
    cv2.line(img, jointPos(v, 5), jointPos(v, 6), c)

    # left leg
    #cv2.line(img, jointPos(v, 4, , , rrr), jointPos(v, 7), c)
    #cv2.line(img, jointPos(v, 7), jointPos(v, 8), c)
    #cv2.line(img, jointPos(v, 8), jointPos(v, 9), c)
    #cv2.line(img, jointPos(v, 9), jointPos(v, 10), c)

    # right leg
    #cv2.line(img, jointPos(v, 1), jointPos(v, 11), c)
    #cv2.line(img, jointPos(v, 11), jointPos(v, 12), c)
    #cv2.line(img, jointPos(v, 12), jointPos(v, 13), c)

    cv2.circle(img, jointPos(v, 0), r, (0, 0, 0), -1)

    cv2.circle(img, jointPos(v, 1), r, (0, 255, 0), -1)
    cv2.circle(img, jointPos(v, 2), r, (100, 255, 0), -1)
    cv2.circle(img, jointPos(v, 3), r, (200, 255, 0), -1)

    cv2.circle(img, jointPos(v, 4), r, (0, 0, 255), -1)
    cv2.circle(img, jointPos(v, 5), r, (100, 0, 255), -1)
    cv2.circle(img, jointPos(v, 6), r, (200, 0, 255), -1)
    
    cv2.circle(img, jointPos(v, 7), r, (255, 255, 255), -1)
    
    cv2.circle(img, jointPos(v, 7), r, (255, 0, 0), -1)
    cv2.circle(img, jointPos(v, 8), r, (255, 127, 0), -1)
    cv2.circle(img, jointPos(v, 9), r, (255, 255, 0), -1)
     
    cv2.circle(img, jointPos(v, 10), r, (255, 0, 255), -1)
    cv2.circle(img, jointPos(v, 11), r, (255, 127, 255), -1)
    cv2.circle(img, jointPos(v, 12), r, (255, 255, 255), -1)

if __name__ == '__main__':
    while True:
        v = np.abs(predict(model, FEATURES))
        vis = visualizeSkeleton(ME, v, SIZE / 64, (0, 255, 0))
        cv2.imshow("Visualization", cv2.resize(vis, (512, 512)))

        if cv2.waitKey(1) == 27:
            break
