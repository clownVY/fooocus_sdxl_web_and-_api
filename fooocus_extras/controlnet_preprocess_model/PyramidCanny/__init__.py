import cv2
import numpy as np
import modules.advanced_parameters as advanced_parameters
from fooocus_extras.controlnet_preprocess_model.ZeoDepth import ZoeDetector


class PyramidCanny:
    # remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
    # modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
    def __init__(self, model_path):
        self.model = None

    @staticmethod
    def centered_canny(x: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2 and x.dtype == np.uint8

        y = cv2.Canny(x, int(advanced_parameters.canny_low_threshold), int(advanced_parameters.canny_high_threshold))
        y = y.astype(np.float32) / 255.0
        return y

    @staticmethod
    def centered_canny_color(x: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 3 and x.shape[2] == 3

        result = [PyramidCanny.centered_canny(x[..., i]) for i in range(3)]
        result = np.stack(result, axis=2)
        return result

    @staticmethod
    def pyramid_canny_color(x: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert x.ndim == 3 and x.shape[2] == 3

        H, W, C = x.shape
        acc_edge = None

        for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            Hs, Ws = int(H * k), int(W * k)
            small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
            edge = PyramidCanny.centered_canny_color(small)
            if acc_edge is None:
                acc_edge = edge
            else:
                acc_edge = cv2.resize(acc_edge, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR)
                acc_edge = acc_edge * 0.75 + edge * 0.25

        return acc_edge

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
        # For some reasons, SAI's Control-lora PyramidCanny seems to be trained on canny maps with non-standard resolutions.
        # Then we use pyramid to use all resolutions to avoid missing any structure in specific resolutions.

        color_canny = PyramidCanny.pyramid_canny_color(RGB)
        result = np.sum(color_canny, axis=2)

        return self.norm255(result, low=1, high=99).clip(0, 255).astype(np.uint8)
