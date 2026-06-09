import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(640,640)
)

def get_embedding(image_path):

    img = cv2.imread(image_path)

    start = time.perf_counter()

    faces = app.get(img)

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    print(
        f"{image_path.split(chr(92))[-1]} : "
        f"{latency_ms:.2f} ms"
    )

    if len(faces) == 0:
        return None

    return faces[0].embedding


img1 = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"

img2 = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Adam_Sandler\Adam_Sandler_0001.jpg"

overall_start = time.perf_counter()

emb1 = get_embedding(img1)
emb2 = get_embedding(img2)

similarity = np.dot(
    emb1,
    emb2
) / (
    np.linalg.norm(emb1)
    * np.linalg.norm(emb2)
)

overall_end = time.perf_counter()

print()
print("Similarity:", similarity)

print(
    "Total Verification Time:",
    round(
        (overall_end-overall_start)*1000,
        2
    ),
    "ms"
)

image_path = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"

img = cv2.imread(image_path)

import time

for i in range(5):

    start = time.perf_counter()

    faces = app.get(img)

    end = time.perf_counter()

    print(
        f"Run {i+1}: {(end-start)*1000:.2f} ms"
    )