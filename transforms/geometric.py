import cv2
import numpy as np
import random
import cv2
import numpy as np
import random
from transforms.base import BaseTransform
class Rotate(BaseTransform):
    def __init__(self, angle=0, mode='random'):
        super().__init__(mode)
        self.angle = angle

    def __call__(self, img):
        h, w = img.shape[:2]
        if self.mode == 'random':
            angle = random.uniform(-self.angle, self.angle)
        else:
            angle = self.angle
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), borderValue=255)


class Translate(BaseTransform):
    def __init__(self, x=0, y=0, mode='random'):
        super().__init__(mode)
        self.x = x
        self.y = y

    def __call__(self, img):
        h, w = img.shape[:2]

        if self.mode == 'random':
            tx = random.randint(-self.x, self.x)
            ty = random.randint(-self.y, self.y)
        else:
            tx = self.x
            ty = self.y

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h), borderValue=255)

class Scale(BaseTransform):
    def __init__(self, min_factor=0.9, max_factor=1.1, mode='random'):
        super().__init__(mode)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img):
        if img is None:
            raise ValueError("Input image is None")
        
        h, w = img.shape[:2]

        # 选择缩放比例
        if self.mode == 'random':
            scale = random.uniform(self.min_factor, self.max_factor)
        else:
            scale = (self.min_factor + self.max_factor) / 2

        # 缩放图像
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 生成 canvas
        canvas = np.full((h, w), 255, dtype=np.uint8)

        # 计算偏移与裁剪
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2

        # 计算放入 canvas 的范围（确保不会越界）
        src_x1 = max(0, -offset_x)
        src_y1 = max(0, -offset_y)
        dst_x1 = max(0, offset_x)
        dst_y1 = max(0, offset_y)

        src_x2 = min(new_w, w - dst_x1 + src_x1)
        src_y2 = min(new_h, h - dst_y1 + src_y1)

        # 粘贴图像到 canvas（可居中、可裁剪）
        canvas[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = \
            resized[src_y1:src_y2, src_x1:src_x2]

        return canvas
