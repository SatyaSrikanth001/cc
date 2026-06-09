from fastapi import FastAPI, UploadFile, File
import shutil
import os

from phase14a import FaceVerifier

app = FastAPI()

verifier = FaceVerifier()

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@app.post("/verify")
async def verify_faces(

    image1: UploadFile = File(...),
    image2: UploadFile = File(...)

):

    path1 = os.path.join(
        UPLOAD_DIR,
        image1.filename
    )

    path2 = os.path.join(
        UPLOAD_DIR,
        image2.filename
    )

    with open(path1, "wb") as f:
        shutil.copyfileobj(
            image1.file,
            f
        )

    with open(path2, "wb") as f:
        shutil.copyfileobj(
            image2.file,
            f
        )

    result = verifier.verify(
        path1,
        path2
    )

    return result