import os
import random
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name='buffalo_s',
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(320,320)
)

LFW_ROOT = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled"


def get_embedding(path):

    img = cv2.imread(path)

    faces = app.get(img)

    if len(faces) == 0:
        return None

    return faces[0].embedding


def similarity(e1, e2):

    return np.dot(
        e1,
        e2
    ) / (
        np.linalg.norm(e1)
        * np.linalg.norm(e2)
    )


people = []

for person in os.listdir(LFW_ROOT):

    folder = os.path.join(
        LFW_ROOT,
        person
    )
    if not os.path.isdir(folder):
        continue
    imgs = [
        os.path.join(folder, x)
        for x in os.listdir(folder)
        if x.endswith(".jpg")
    ]

    if len(imgs) >= 2:
        people.append(imgs)

print(
    "People with >=2 images:",
    len(people)
)