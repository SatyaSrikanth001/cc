def _extract_swipe_geometry(self, touches, screen_w, screen_h):
    angles = []
    straightness_vals = []
    path_lengths = []

    for t in touches:
        # If raw touch-move points are available (e.g., t['points']), use them;
        # otherwise approximate using touch-down/up coordinates.
        x1 = t.get('x_down', t.get('x')) / screen_w
        y1 = t.get('y_down', t.get('y')) / screen_h
        x2 = t.get('x_up', t.get('x')) / screen_w
        y2 = t.get('y_up', t.get('y')) / screen_h
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        angles.append(angle)
        path_len = np.sqrt(dx**2 + dy**2)
        path_lengths.append(path_len)
        # Straightness = direct distance / sum of segment lengths (approximate as 1 if only two points)
        straightness_vals.append(1.0)  # or compute from raw points if available

    features = {}
    if angles:
        features['swipe_angle_mean'] = np.mean(angles)
        features['swipe_angle_std'] = np.std(angles)
        features['swipe_path_mean'] = np.mean(path_lengths)
        features['swipe_path_std'] = np.std(path_lengths)
        features['swipe_straightness_mean'] = np.mean(straightness_vals)
    else:
        # default zeros
        for k in ['swipe_angle_mean','swipe_angle_std','swipe_path_mean','swipe_path_std','swipe_straightness_mean']:
            features[k] = 0.0
    return features