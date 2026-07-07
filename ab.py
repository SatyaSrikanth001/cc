import json
with open("data/new_raw/<user>.json") as f:
    data = json.load(f)
sessions = data['data']['sessions']
for i, s in enumerate(sessions):
    t = s.get('touch_events', [])
    se = s.get('sensor_events', [])
    print(f"Session {i} ({s.get('session_id')}): touches={len(t)}, sensors={len(se)}")
    if t:
        sample = t[0]
        print(f"   sample touch keys: {list(sample.keys())}, down_ts={sample.get('touch_down_ts')}, up_ts={sample.get('touch_up_ts')}")
    if se:
        sample_s = se[0]
        print(f"   sample sensor keys: {list(sample_s.keys())}")
