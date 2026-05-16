class BBoxSmoother:
    def __init__(self, alpha: float = 0.65):
        self.alpha = alpha
        self.bbox = None

    def update(self, bbox):
        if bbox is None:
            return self.bbox
        if self.bbox is None:
            self.bbox = bbox
            return bbox
        self.bbox = tuple(
            int(self.alpha * old + (1.0 - self.alpha) * new)
            for old, new in zip(self.bbox, bbox)
        )
        return self.bbox

    def reset(self):
        self.bbox = None
