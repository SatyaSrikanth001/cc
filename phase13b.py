import os
import random
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
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


def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
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

print("People:", len(people))

same_scores = []
different_scores = []

# ------------------
# SAME PERSON PAIRS
# ------------------

for _ in range(100):

    imgs = random.choice(people)

    img1, img2 = random.sample(imgs, 2)

    e1 = get_embedding(img1)
    e2 = get_embedding(img2)

    if e1 is None or e2 is None:
        continue

    score = cosine_similarity(e1, e2)

    same_scores.append(score)

# ------------------
# DIFFERENT PAIRS
# ------------------

for _ in range(100):

    person1, person2 = random.sample(
        people,
        2
    )

    img1 = random.choice(person1)
    img2 = random.choice(person2)

    e1 = get_embedding(img1)
    e2 = get_embedding(img2)

    if e1 is None or e2 is None:
        continue

    score = cosine_similarity(e1, e2)

    different_scores.append(score)

print()
print("Same Pair Avg:",
      np.mean(same_scores))

print("Different Pair Avg:",
      np.mean(different_scores))

print()

print("Same Min:",
      np.min(same_scores))

print("Same Max:",
      np.max(same_scores))

print()

print("Different Min:",
      np.min(different_scores))

print("Different Max:",
      np.max(different_scores))