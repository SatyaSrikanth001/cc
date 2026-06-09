import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis


class FaceVerifier:

    def __init__(self):

        self.app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(320,320)
        )

    def get_embedding(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            raise Exception(
                f"Cannot read image: {image_path}"
            )

        faces = self.app.get(img)

        if len(faces) == 0:
            raise Exception(
                f"No face found: {image_path}"
            )

        return faces[0].embedding

    def cosine_similarity(
        self,
        emb1,
        emb2
    ):

        return float(
            np.dot(emb1, emb2)
            /
            (
                np.linalg.norm(emb1)
                *
                np.linalg.norm(emb2)
            )
        )

    def verify(
        self,
        image1_path,
        image2_path,
        threshold=0.25
    ):

        start = time.perf_counter()

        emb1 = self.get_embedding(
            image1_path
        )

        emb2 = self.get_embedding(
            image2_path
        )

        score = self.cosine_similarity(
            emb1,
            emb2
        )

        end = time.perf_counter()

        latency_ms = (
            end-start
        ) * 1000

        return {
            "similarity": score,
            "same_person": score >= threshold,
            "threshold": threshold,
            "latency_ms": round(
                latency_ms,
                2
            )
        }