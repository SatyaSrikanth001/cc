from phase14a import FaceVerifier

verifier = FaceVerifier()

img1 = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"

img2 = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Adam_Sandler\Adam_Sandler_0001.jpg"

result = verifier.verify(
    img1,
    img2
)

print(result)