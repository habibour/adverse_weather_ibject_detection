"""
Atmospheric-scattering-model (ASM) fog synthesis, following IA-YOLO's
data_make.py (https://github.com/wenyyu/Image-Adaptive-YOLO/blob/main/core/data_make.py),
which is the method RDFNet's paper cites for building VOC-FOG
(Sec. IV-1: "we build a VOC-FOG dataset ... following the approach of IA-YOLO").

I(x) = J(x) * t(x) + A * (1 - t(x)),      t(x) = exp(-beta * d(x))

d(x) is NOT a real depth map -- IA-YOLO uses a radial "distance from image
center" proxy so fog can be synthesized offline with no depth network:
    d(row, col) = -0.04 * sqrt((row - h/2)^2 + (col - w/2)^2) + sqrt(max(h, w))
A = 0.5 (atmospheric light)
beta: IA-YOLO's original code creates 10 fixed levels, beta = 0.05, 0.06, ..., 0.14
      (one image -> ten foggy copies). RDFNet's dataset_split/train.txt and
      test.txt each list exactly one entry per source photo, so here we sample
      a single beta ~ Uniform(0.05, 0.14) per image instead of emitting all ten.
"""
import math
import numpy as np


def add_fog(image_bgr_uint8, beta=None, A=0.5, rng=None):
    """
    image_bgr_uint8: HxWx3 uint8 array (as returned by cv2.imread)
    beta: fog density coefficient; sampled from Uniform(0.05, 0.14) if None
    rng: a random.Random or np.random.RandomState/Generator instance (optional,
         pass one in for reproducible per-image seeding)
    Returns (foggy_uint8, beta_used)
    """
    rng = rng if rng is not None else np.random
    if beta is None:
        beta = rng.uniform(0.05, 0.14)

    h, w = image_bgr_uint8.shape[:2]
    img = image_bgr_uint8.astype(np.float32) / 255.0
    size = math.sqrt(max(h, w))
    cy, cx = h // 2, w // 2

    yy, xx = np.mgrid[0:h, 0:w]
    d = -0.04 * np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) + size
    td = np.exp(-beta * d).astype(np.float32)[..., None]

    foggy = img * td + A * (1 - td)
    foggy = np.clip(foggy * 255.0, 0, 255).astype(np.uint8)
    return foggy, beta
