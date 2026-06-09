import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(640,640)
)

img = cv2.imread(
    r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"
)

if img is None:
    print("IMAGE NOT LOADED")
    exit()

faces = app.get(img)

print("Faces found:", len(faces))

if len(faces) > 0:
    print("Embedding shape:", faces[0].embedding.shape)