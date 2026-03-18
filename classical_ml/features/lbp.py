import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.transform import resize

def extract_lbp(img_path, P=8, R=1):
    img = imread(img_path)

    # Resize to match HOG preprocessing
    img = resize(img, (150, 150), anti_aliasing=True)

    if img.ndim == 3:
        img = rgb2gray(img)

    # Convert to uint8 safely
    if img.dtype != np.uint8:
        img = (img * 255).astype("uint8")

    lbp = local_binary_pattern(img, P=P, R=R, method="uniform")

    # For uniform LBP, bins = P + 2
    n_bins = P + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist