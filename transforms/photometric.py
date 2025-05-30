import numpy as np
import cv2

class GaussianNoise:
    def __init__(self, mean=0, std=10, mode='random'):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
