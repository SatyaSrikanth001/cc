import cv2
import time
import statistics
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name='buffalo_s',
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(320,320)
)

img = cv2.imread(
    r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg"
)

times = []

# warmup
app.get(img)

for i in range(50):

    start = time.perf_counter()

    app.get(img)

    end = time.perf_counter()

    latency = (end-start)*1000

    times.append(latency)

avg = statistics.mean(times)

times.sort()

p50 = times[int(len(times)*0.50)]
p95 = times[int(len(times)*0.95)]
p99 = times[int(len(times)*0.99)]

print(f"Average: {avg:.2f} ms")
print(f"P50: {p50:.2f} ms")
print(f"P95: {p95:.2f} ms")
print(f"P99: {p99:.2f} ms")