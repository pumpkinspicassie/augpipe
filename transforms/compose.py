import random

class ComposeTransform:
    def __init__(self, transforms, mode="fixed"):
        self.transforms = []
        for t in transforms:
            if isinstance(t, tuple):
                transform, p = t
            else:
                transform, p = t, 1.0
            self.transforms.append((transform, p))
        self.mode = mode

    def __call__(self, img):
        for t, p in self.transforms:
            if hasattr(t, "mode"):
                t.mode = self.mode
            if random.random() < p:
                img = t(img)
        return img

class OneOfTransform:
    def __init__(self, transforms, mode="fixed"):
        """
        Randomly choose ONE transform to apply from a list.

        :param transforms: list of (transform, p) tuples OR just transform objects
        :param mode: passed to each transform if applicable
        """
        self.transforms = []
        total_p = 0.0
        for t in transforms:
            if isinstance(t, tuple):
                transform, p = t
            else:
                transform, p = t, 1.0
            total_p += p
            self.transforms.append((transform, p))

        self.mode = mode
        self.total_p = total_p

    def __call__(self, img):
        r = random.random() * self.total_p
        cumulative = 0.0
        for t, p in self.transforms:
            cumulative += p
            if r < cumulative:
                if hasattr(t, "mode"):
                    t.mode = self.mode
                return t(img)
        return img

class SometimesTransform:
    def __init__(self, transform, p=0.5, mode="fixed"):
        """
        Apply the given transform with probability p.

        :param transform: a callable transform
        :param p: probability of applying the transform
        :param mode: passed to the transform if applicable
        """
        self.transform = transform
        self.p=p
        self.mode = mode

    def __call__(self, img):
        if hasattr(self.transform, "mode"):
            self.transform.mode = self.mode
        if random.random() < self.p:
            return self.transform(img)
        return img
