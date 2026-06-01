import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from classical_ml.features.hog import extract_hog, extract_hog_features, save_features
from classical_ml.features.lbp import extract_lbp


def _save_image(path, array):
    Image.fromarray(array).save(path)


class TestHogExtraction(unittest.TestCase):
    def test_extract_hog_from_grayscale_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "gray.png")
            gray = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
            _save_image(img_path, gray)

            features = extract_hog(img_path)

            self.assertIsInstance(features, np.ndarray)
            self.assertGreater(features.size, 0)
            self.assertEqual(features.ndim, 1)

    def test_extract_hog_from_rgba_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "rgba.png")
            rgba = np.random.randint(0, 255, size=(64, 64, 4), dtype=np.uint8)
            _save_image(img_path, rgba)

            features = extract_hog(img_path)

            self.assertIsInstance(features, np.ndarray)
            self.assertGreater(features.size, 0)
            self.assertEqual(features.ndim, 1)

    def test_extract_hog_features_skips_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class_names = ["class_a", "class_b"]
            for class_name in class_names:
                os.makedirs(os.path.join(tmpdir, class_name), exist_ok=True)

            _save_image(
                os.path.join(tmpdir, "class_a", "a1.png"),
                np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
            )
            _save_image(
                os.path.join(tmpdir, "class_b", "b1.png"),
                np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
            )

            with open(os.path.join(tmpdir, "class_a", "broken.txt"), "w", encoding="utf-8") as f:
                f.write("not an image")

            X, y = extract_hog_features(tmpdir, class_names)

            self.assertEqual(X.shape[0], 2)
            self.assertEqual(y.shape[0], 2)
            self.assertSetEqual(set(y.tolist()), {0, 1})

    def test_save_features_writes_expected_npy_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            X = np.random.rand(3, 5)
            y = np.array([0, 1, 2])

            from classical_ml.features import hog as hog_module

            original_feature_dir = hog_module.feature_dir
            hog_module.feature_dir = tmpdir
            try:
                save_features(X, y, "sample")
            finally:
                hog_module.feature_dir = original_feature_dir

            X_saved = np.load(os.path.join(tmpdir, "sample_X.npy"))
            y_saved = np.load(os.path.join(tmpdir, "sample_y.npy"))

            self.assertTrue(np.allclose(X, X_saved))
            self.assertTrue(np.array_equal(y, y_saved))


class TestLbpExtraction(unittest.TestCase):
    def test_extract_lbp_default_histogram_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "rgb.png")
            rgb = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
            _save_image(img_path, rgb)

            hist = extract_lbp(img_path)

            self.assertIsInstance(hist, np.ndarray)
            self.assertEqual(hist.shape[0], 10)
            self.assertAlmostEqual(hist.sum(), 1.0, places=5)
            self.assertTrue(np.all(hist >= 0))

    def test_extract_lbp_custom_bin_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "rgb_custom.png")
            rgb = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
            _save_image(img_path, rgb)

            hist = extract_lbp(img_path, P=16, R=2)

            self.assertEqual(hist.shape[0], 18)
            self.assertAlmostEqual(hist.sum(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
