import torch
import os
from einops import rearrange
import numpy as np
import cv2


class CPDS:
    # remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
    # modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
    def __init__(self, model_path):
        self.model = None

    @classmethod
    def norm255(cls, GrayImage, low=4, high=96):
        assert isinstance(GrayImage, np.ndarray)
        assert GrayImage.ndim == 2 and GrayImage.dtype == np.float32

        v_min = np.percentile(GrayImage, low)
        v_max = np.percentile(GrayImage, high)

        if np.allclose(v_min, v_max):
            GrayImage = GrayImage * 0  # Avoid 0-division
        else:
            GrayImage = (GrayImage - v_min) / (v_max - v_min)

        GrayImage -= v_min
        GrayImage /= v_max - v_min
        return GrayImage * 255.0

    def __call__(self, RGB):
        assert RGB.ndim == 3
        with torch.no_grad():
            # cv2.decolor is not "decolor", it is Cewu Lu's method
            # See http://www.cse.cuhk.edu.hk/leojia/projects/color2gray/index.html
            # See https://docs.opencv.org/3.0-beta/modules/photo/doc/decolor.html

            raw = cv2.GaussianBlur(RGB, (0, 0), 0.8)
            density, boost = cv2.decolor(raw)

            raw = raw.astype(np.float32)
            density = density.astype(np.float32)
            boost = boost.astype(np.float32)

            offset = np.sum((raw - boost) ** 2.0, axis=2) ** 0.5
            result = density + offset

            return self.norm255(result, low=4, high=96).clip(0, 255).astype(np.uint8)
