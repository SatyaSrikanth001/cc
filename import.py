import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_reaction_time_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Reaction Time Distribution features from navigation and touch events.
        
        Reaction time is the interval between a screen transition (navigation event)
        and the subsequent touch event. The distribution of these times reflects
        the user's cognitive processing speed and motor preparation, which are
        stable individual traits.
        
        References:
        - Hick WE. (1952). "On the rate of gain of information."
          Q J Exp Psychol. 4(1):11-26.
        - Hyman R. (1953). "Stimulus information as a determinant of reaction time."
          J Exp Psychol. 45(3):188-196.
        - Voss A, et al. (2020). "Differentiating between the effects of response
          time and cognitive ability on biometric identification."
          Behav Res Methods. 52:1887-1902.
        
        Args:
            session: Session dictionary containing touch_events and navigation_events
            
        Returns:
            Dictionary with Reaction Time Distribution features:
            - reaction_time_mean: Mean reaction time (ms)
            - reaction_time_std: Standard deviation (ms)
            - reaction_time_skew: Skewness of the distribution
            - reaction_time_kurtosis: Kurtosis of the distribution
            - reaction_time_p5: 5th percentile (ms)
            - reaction_time_p95: 95th percentile (ms)
            - reaction_time_iqr: Interquartile range (ms)
            - reaction_time_n: Number of valid reaction times
        """
        touches = session.get('touch_events', [])
        navigations = session.get('navigation_events', [])
        
        if not navigations or not touches:
            return self._reaction_time_defaults()
        
        # Sort navigation events by timestamp
        navs = sorted(navigations, key=lambda e: e.get('timestamp', 0))
        # Filter out navigation events without timestamp
        navs = [n for n in navs if 'timestamp' in n and n['timestamp'] is not None]
        if not navs:
            return self._reaction_time_defaults()
        
        # Get all touch down timestamps, sorted
        touch_times = sorted([t['touch_down_ts'] for t in touches if 'touch_down_ts' in t])
        if not touch_times:
            return self._reaction_time_defaults()
        
        reaction_times = []
        touch_idx = 0
        n_touches = len(touch_times)
        
        for nav in navs:
            nav_ts = nav['timestamp']
            # Find the first touch that occurs after this navigation
            while touch_idx < n_touches and touch_times[touch_idx] <= nav_ts:
                touch_idx += 1
            if touch_idx >= n_touches:
                break
            rt = touch_times[touch_idx] - nav_ts
            # Filter realistic reaction times (100 ms to 5 seconds)
            if 100 <= rt <= 5000:
                reaction_times.append(rt)
            # Move to next touch for subsequent navs? We should not consume the touch
            # because a touch may be the response to only one nav; we should find the
            # first touch after each nav, but a touch can only be the response to
            # the closest previous nav. So we need to ensure we don't skip touches
            # that might be responses to multiple navs? Actually, each touch is a
            # response to some event; we should assign each touch to the most recent
            # navigation that occurred before it. A simpler approach: for each touch,
            # find the most recent navigation that occurred before it, and compute RT.
            # This avoids the issue of skipping touches.
        
        # Better: For each touch, find the most recent navigation before it.
        reaction_times = []
        touch_idx = 0
        nav_ts_list = [n['timestamp'] for n in navs]
        for touch_ts in touch_times:
            # Find the most recent navigation timestamp <= touch_ts
            # Use binary search
            import bisect
            idx = bisect.bisect_right(nav_ts_list, touch_ts) - 1
            if idx >= 0:
                rt = touch_ts - nav_ts_list[idx]
                if 100 <= rt <= 5000:
                    reaction_times.append(rt)
        
        if len(reaction_times) < 3:
            return self._reaction_time_defaults()
        
        rt_array = np.array(reaction_times)
        mean_rt = np.mean(rt_array)
        std_rt = np.std(rt_array)
        skew_rt = stats.skew(rt_array)
        kurt_rt = stats.kurtosis(rt_array)  # excess kurtosis
        p5 = np.percentile(rt_array, 5)
        p95 = np.percentile(rt_array, 95)
        q1 = np.percentile(rt_array, 25)
        q3 = np.percentile(rt_array, 75)
        iqr = q3 - q1
        
        return {
            'reaction_time_mean': float(mean_rt),
            'reaction_time_std': float(std_rt),
            'reaction_time_skew': float(skew_rt),
            'reaction_time_kurtosis': float(kurt_rt),
            'reaction_time_p5': float(p5),
            'reaction_time_p95': float(p95),
            'reaction_time_iqr': float(iqr),
            'reaction_time_n': float(len(reaction_times)),
        }
    
    def _reaction_time_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'reaction_time_mean': 0.0,
            'reaction_time_std': 0.0,
            'reaction_time_skew': 0.0,
            'reaction_time_kurtosis': 0.0,
            'reaction_time_p5': 0.0,
            'reaction_time_p95': 0.0,
            'reaction_time_iqr': 0.0,
            'reaction_time_n': 0.0,
        }
































import numpy as np
from typing import Dict, Any, List, Optional

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_cognitive_load_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Cognitive Load Score features from touch and navigation events.
        
        Cognitive Load Score (CLS) measures the mental effort required during
        interaction. Higher cognitive load manifests as:
        - Increased hesitation (longer pauses between actions)
        - Increased timing variability (less consistent performance)
        - Increased corrections (more errors and corrective actions)
        
        The CLS is a weighted composite of three indicators:
            CLS = w_H * hesitation_rate + w_V * variability_index + w_C * correction_rate
        
        References:
        - Sweller J. (1988). "Cognitive load during problem solving."
        - Vlahos M, et al. (2026). "Integrated Research on Cognitive Load,
          Emotions, and Keystroke-Mouse Dynamics." Springer.
        - Study on Cognitive Load Detection using Digital Phenotypes based on
          Smartphone Typing Patterns and Accelerometer Data. (2024).
        
        Args:
            session: Session dictionary containing touch_events and navigation_events
            
        Returns:
            Dictionary with Cognitive Load features:
            - cognitive_load_score: Composite cognitive load score
            - cognitive_load_hesitation: Hesitation rate (pauses > 500ms)
            - cognitive_load_variability: Variability index (CV of durations)
            - cognitive_load_corrections: Correction rate
            - cognitive_load_n_touches: Number of touches used
        """
        touches = session.get('touch_events', [])
        navigations = session.get('navigation_events', [])
        
        if len(touches) < 5:
            return self._cognitive_load_defaults()
        
        # Filter for valid touches with required fields
        valid_touches = [
            t for t in touches
            if all(k in t for k in ['touch_down_ts', 'touch_up_ts'])
        ]
        
        if len(valid_touches) < 5:
            return self._cognitive_load_defaults()
        
        # Extract touch durations
        durations = np.array([
            t['touch_up_ts'] - t['touch_down_ts']
            for t in valid_touches
        ])
        
        # Filter out unrealistic durations (too short or too long)
        durations = durations[(durations >= 20) & (durations <= 5000)]
        
        if len(durations) < 3:
            return self._cognitive_load_defaults()
        
        # Extract inter-touch intervals
        down_times = np.array([t['touch_down_ts'] for t in valid_touches])
        intervals = np.diff(down_times)
        
        # Filter out unrealistic intervals
        intervals = intervals[(intervals >= 10) & (intervals <= 10000)]
        
        # ============================================================
        # 1. HESITATION RATE
        # ============================================================
        # Proportion of inter-touch intervals exceeding 500ms
        hesitation_threshold = 500  # ms
        if len(intervals) > 0:
            hesitation_rate = np.mean(intervals > hesitation_threshold)
        else:
            hesitation_rate = 0.0
        
        # ============================================================
        # 2. VARIABILITY INDEX
        # ============================================================
        # Coefficient of variation of touch durations
        duration_mean = np.mean(durations)
        duration_std = np.std(durations)
        if duration_mean > 0:
            variability_index = duration_std / duration_mean
        else:
            variability_index = 0.0
        
        # ============================================================
        # 3. CORRECTION RATE
        # ============================================================
        # Detect corrections from gesture types and navigation reversals
        corrections = 0
        total_actions = len(valid_touches)
        
        # 3a. Gesture-based corrections
        for t in valid_touches:
            gesture = t.get('gesture_type', '')
            # Swipes might indicate corrections (e.g., scrolling back)
            # Taps might indicate corrections (e.g., retyping)
            if gesture in ['swipe', 'scroll']:
                corrections += 1
        
        # 3b. Navigation-based corrections (back navigation)
        if len(navigations) >= 2:
            expected_flow = [
                "Start Screen", "Register Screen", "Login-Signup Screen",
                "Home Screen", "Payment Screen"
            ]
            for i in range(1, len(navigations)):
                prev_screen = navigations[i-1].get('to_screen', '')
                curr_screen = navigations[i].get('to_screen', '')
                # Check if this is a backward navigation
                if prev_screen in expected_flow and curr_screen in expected_flow:
                    if expected_flow.index(curr_screen) < expected_flow.index(prev_screen):
                        corrections += 1
        
        correction_rate = corrections / (total_actions + 1e-6)
        
        # ============================================================
        # 4. COMPOSITE COGNITIVE LOAD SCORE
        # ============================================================
        # Weights (empirically determined)
        w_h = 0.4   # Hesitation
        w_v = 0.3   # Variability
        w_c = 0.3   # Corrections
        
        # Clamp indicators to [0, 1] for interpretability
        hesitation_rate = np.clip(hesitation_rate, 0.0, 1.0)
        variability_index = np.clip(variability_index, 0.0, 2.0)  # CV can exceed 1
        variability_index = np.clip(variability_index / 2.0, 0.0, 1.0)  # Normalize to [0,1]
        correction_rate = np.clip(correction_rate, 0.0, 1.0)
        
        cognitive_load_score = (
            w_h * hesitation_rate +
            w_v * variability_index +
            w_c * correction_rate
        )
        
        return {
            'cognitive_load_score': float(cognitive_load_score),
            'cognitive_load_hesitation': float(hesitation_rate),
            'cognitive_load_variability': float(variability_index),
            'cognitive_load_corrections': float(correction_rate),
            'cognitive_load_n_touches': float(len(valid_touches)),
        }
    
    def _cognitive_load_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'cognitive_load_score': 0.0,
            'cognitive_load_hesitation': 0.0,
            'cognitive_load_variability': 0.0,
            'cognitive_load_corrections': 0.0,
            'cognitive_load_n_touches': 0.0,
        }































import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, Any, Optional, List, Tuple

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_minimum_jerk_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Minimum Jerk Deviation features from touch events.
        
        This feature quantifies how much the user's actual touch trajectory
        deviates from the optimal minimum-jerk trajectory (Flash & Hogan, 1985).
        The minimum jerk trajectory is the smoothest possible movement between
        two points given a fixed movement duration.
        
        Actual jerk cost is computed from the interpolated touch trajectory,
        and compared to the theoretical minimum jerk cost.
        
        References:
        - Flash T, Hogan N. (1985). "The coordination of arm movements:
          an experimentally confirmed mathematical model." J Neurosci. 5(7):1688-1703.
        - Hogan N, Sternad D. (2009). "Sensitivity of smoothness measures to
          movement duration, amplitude, and arrests." J Mot Behav. 41(6):529-534.
        
        Args:
            session: Session dictionary containing touch_events
            
        Returns:
            Dictionary with Minimum Jerk Deviation features:
            - min_jerk_deviation_mean: Mean squared positional deviation (normalized)
            - min_jerk_deviation_max: Maximum positional deviation (normalized)
            - min_jerk_cost_ratio: Ratio of actual jerk cost to minimum jerk cost
            - min_jerk_n_points: Number of touch points used
        """
        touches = session.get('touch_events', [])
        if len(touches) < 3:
            return self._min_jerk_defaults()
        
        # Extract touch positions and times (use touch_down_ts)
        points = []
        for t in touches:
            if all(k in t for k in ['x', 'y', 'touch_down_ts']):
                points.append((t['touch_down_ts'], t['x'], t['y']))
        if len(points) < 3:
            return self._min_jerk_defaults()
        
        # Normalize coordinates by screen size
        screen_w = session.get('device_info', {}).get('screen_width', 1)
        screen_h = session.get('device_info', {}).get('screen_height', 1)
        if screen_w == 0 or screen_h == 0:
            screen_w, screen_h = 1, 1
        
        # Sort by time
        points = sorted(points, key=lambda p: p[0])
        times = np.array([p[0] for p in points])
        xs = np.array([p[1] / screen_w for p in points])
        ys = np.array([p[2] / screen_h for p in points])
        
        # Total duration
        T = times[-1] - times[0]
        if T <= 0:
            return self._min_jerk_defaults()
        
        # Interpolate to uniform grid (e.g., 100 Hz)
        fs = 100.0
        dt = 1.0 / fs
        num_samples = int(np.ceil(T * fs)) + 1
        t_uniform = np.linspace(times[0], times[-1], num_samples)
        
        # Interpolate x and y (use cubic spline for smoothness)
        try:
            f_x = interp1d(times, xs, kind='cubic', fill_value='extrapolate')
            f_y = interp1d(times, ys, kind='cubic', fill_value='extrapolate')
            x_interp = f_x(t_uniform)
            y_interp = f_y(t_uniform)
        except Exception:
            # Fallback to linear
            f_x = interp1d(times, xs, kind='linear', fill_value='extrapolate')
            f_y = interp1d(times, ys, kind='linear', fill_value='extrapolate')
            x_interp = f_x(t_uniform)
            y_interp = f_y(t_uniform)
        
        # Compute derivatives using finite differences
        vx = np.gradient(x_interp, dt)
        vy = np.gradient(y_interp, dt)
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        jerk_x = np.gradient(ax, dt)
        jerk_y = np.gradient(ay, dt)
        
        # Actual jerk cost (integral of squared jerk)
        J_actual = np.sum(jerk_x**2 + jerk_y**2) * dt
        
        # Minimum jerk trajectory and its cost
        x0, xf = xs[0], xs[-1]
        y0, yf = ys[0], ys[-1]
        distance = np.sqrt((xf - x0)**2 + (yf - y0)**2)
        
        # Compute minimum jerk cost analytically: J_min = 360 * distance^2 / T^5
        if distance < 1e-12 or T <= 0:
            return self._min_jerk_defaults()
        J_min = 360.0 * (distance**2) / (T**5)
        
        # Compute minimum jerk trajectory (for deviation)
        tau = (t_uniform - times[0]) / T
        # Clamp tau to [0,1]
        tau = np.clip(tau, 0, 1)
        x_opt = x0 + (xf - x0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
        y_opt = y0 + (yf - y0) * (10*tau**3 - 15*tau**4 + 6*tau**5)
        
        # Positional deviation (mean squared error normalized by distance^2)
        dx = x_interp - x_opt
        dy = y_interp - y_opt
        squared_error = dx**2 + dy**2
        mean_sq_err = np.mean(squared_error)
        max_err = np.max(np.sqrt(squared_error))
        # Normalize by distance^2 for mean squared error, by distance for max
        if distance > 0:
            norm_mean = mean_sq_err / (distance**2)
            norm_max = max_err / distance
        else:
            norm_mean = 0.0
            norm_max = 0.0
        
        # Jerk cost ratio
        if J_min > 1e-12:
            cost_ratio = J_actual / J_min
        else:
            cost_ratio = 0.0
        
        return {
            'min_jerk_deviation_mean': float(norm_mean),
            'min_jerk_deviation_max': float(norm_max),
            'min_jerk_cost_ratio': float(cost_ratio),
            'min_jerk_n_points': float(len(points)),
        }
    
    def _min_jerk_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'min_jerk_deviation_mean': 0.0,
            'min_jerk_deviation_max': 0.0,
            'min_jerk_cost_ratio': 0.0,
            'min_jerk_n_points': 0.0,
        }