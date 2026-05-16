from collections import deque
import numpy as np


class TemporalRGBBuffer:
    def __init__(self, maxlen: int):
        self.values = deque(maxlen=maxlen)

    def append(self, value):
        if value is not None and len(value) == 9:
            self.values.append(np.asarray(value, dtype=np.float32))

    def clear(self):
        self.values.clear()

    def ready(self):
        return len(self.values) == self.values.maxlen

    def array(self):
        return np.stack(self.values, axis=0)

    def padded_array(self):
        if not self.values:
            return np.zeros((self.values.maxlen, 9), dtype=np.float32)
        x = np.stack(self.values, axis=0)
        if len(x) >= self.values.maxlen:
            return x[-self.values.maxlen :]
        pad = np.repeat(x[:1], self.values.maxlen - len(x), axis=0)
        return np.concatenate([pad, x], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.values)

    def fill_ratio(self):
        return len(self.values) / float(self.values.maxlen)


def estimate_signal_quality(x: np.ndarray, fps: float):
    if x is None or len(x) < max(12, int(fps)):
        return {"pulse_consistency": 0.0, "estimated_hr": 0.0, "motion_quality": 0.0}
    green = x[:, [1, 4, 7]].mean(axis=1)
    green = green - green.mean()
    diff_energy = float(np.mean(np.abs(np.diff(green)))) + 1e-6
    signal_energy = float(np.std(green)) + 1e-6
    consistency = float(np.clip(signal_energy / (signal_energy + 2.5 * diff_energy), 0.0, 1.0))

    freqs = np.fft.rfftfreq(len(green), d=1.0 / max(fps, 1.0))
    spectrum = np.abs(np.fft.rfft(green))
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if mask.any() and spectrum[mask].max() > 0:
        peak_freq = freqs[mask][np.argmax(spectrum[mask])]
        hr = float(peak_freq * 60.0)
    else:
        hr = 0.0
    motion_quality = float(np.clip(1.0 - diff_energy * 30.0, 0.0, 1.0))
    return {"pulse_consistency": consistency, "estimated_hr": hr, "motion_quality": motion_quality}
