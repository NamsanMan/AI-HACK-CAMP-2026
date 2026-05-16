class ScoreSmoother:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.value = None

    def update(self, raw: float) -> float:
        raw = float(raw)
        if self.value is None:
            self.value = raw
        else:
            self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        return self.value


class RiskHysteresis:
    def __init__(
        self,
        suspicious_enter: float = 0.65,
        suspicious_exit: float = 0.55,
        high_enter: float = 0.85,
        high_exit: float = 0.75,
    ):
        self.suspicious_enter = suspicious_enter
        self.suspicious_exit = suspicious_exit
        self.high_enter = high_enter
        self.high_exit = high_exit
        self.state = "Low"

    def reset(self, state: str = "Low"):
        self.state = state
        return self.state

    def update(self, score: float, verified: bool = True, allow_high: bool = True) -> str:
        if not verified:
            self.state = "Unverified"
            return self.state
        score = float(score)
        if self.state == "Unverified":
            self.state = "Low"
        if self.state == "High":
            if score <= self.high_exit:
                self.state = "Suspicious" if score >= self.suspicious_exit else "Low"
        elif self.state == "Suspicious":
            if allow_high and score >= self.high_enter:
                self.state = "High"
            elif score <= self.suspicious_exit:
                self.state = "Low"
        else:
            if allow_high and score >= self.high_enter:
                self.state = "High"
            elif score >= self.suspicious_enter:
                self.state = "Suspicious"
        return self.state


def risk_color_bgr(state: str):
    if state == "Unverified":
        return (170, 170, 170)
    if state == "High":
        return (40, 40, 255)
    if state == "Suspicious":
        return (40, 180, 255)
    return (80, 220, 80)


def risk_label(state: str):
    if state == "Unverified":
        return "Unverified"
    if state == "High":
        return "High Risk"
    if state == "Suspicious":
        return "Watch"
    return "Low Risk"
