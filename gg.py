import json
import numpy as np

with open("one_user.json") as f:
    data = json.load(f)

session = data["data"]["sessions"]["sessions"][0]

timestamps = [
    e["timestamp"]
    for e in session["sensor_events"]
]

print("events =", len(timestamps))
print("unique =", len(set(timestamps)))