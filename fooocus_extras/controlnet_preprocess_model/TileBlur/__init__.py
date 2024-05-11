import cv2


class TileBlur:
    def __init__(self,model_path):
        self.model = None

    def __call__(self,RGB):
        assert RGB.ndim == 3
        assert RGB.shape[2] == 3
        RGB = cv2.GaussianBlur(RGB, (3, 3), 3)
        return RGB