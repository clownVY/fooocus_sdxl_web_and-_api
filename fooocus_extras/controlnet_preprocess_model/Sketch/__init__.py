import cv2


class Sketch:
    def __init__(self, model_path):
        self.model = None

    def __call__(self, RGB):
        assert RGB.ndim == 3
        assert RGB.shape[2] == 3
        Gray = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)
        return Gray
