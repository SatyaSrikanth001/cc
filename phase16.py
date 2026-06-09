import requests
import time
import statistics

URL = "http://127.0.0.1:8000/verify"

img1_path = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Abdullatif_Sener\Abdullatif_Sener_0001.jpg"

img2_path = r"C:\Users\LOLLA SRIKANTH\scikit_learn_data\lfw_home\lfw_funneled\Abdullatif_Sener\Abdullatif_Sener_0002.jpg"

times = []

for i in range(20):

    with open(img1_path, "rb") as f1, \
         open(img2_path, "rb") as f2:

        start = time.perf_counter()

        response = requests.post(
            URL,
            files={
                "image1": f1,
                "image2": f2
            }
        )

        end = time.perf_counter()

        latency = (
            end-start
        ) * 1000

        times.append(latency)

        print(
            f"Run {i+1}: {latency:.2f} ms"
        )

avg = statistics.mean(times)

times.sort()

p50 = times[int(len(times)*0.50)]
p95 = times[int(len(times)*0.95)]

print()
print(f"Average: {avg:.2f} ms")
print(f"P50: {p50:.2f} ms")
print(f"P95: {p95:.2f} ms")