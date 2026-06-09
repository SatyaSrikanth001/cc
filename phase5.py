import cv2
import numpy as np
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

    faces = app.get(img)

    if len(faces) == 0:
        return None

    return faces[0].embedding


img1 = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"

# img2 = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"
img2 = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Adam_Sandler\Adam_Sandler_0001.jpg"

emb1 = get_embedding(img1)
emb2 = get_embedding(img2)

similarity = np.dot(
    emb1,
    emb2
) / (
    np.linalg.norm(emb1)
    * np.linalg.norm(emb2)
)

print("Similarity:", similarity)