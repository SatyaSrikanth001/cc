import cv2
import time
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name='buffalo_s',
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(640,640)
)

img = cv2.imread(
    r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"
)

for i in range(5):

    start = time.perf_counter()

    faces = app.get(img)

    end = time.perf_counter()

    print(
        f"Run {i+1}: {(end-start)*1000:.2f} ms"
    )