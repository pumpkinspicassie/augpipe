# augpipe

---

## Functions

- Geometric:Rotate, Scale, Translate; Damage:BlackDropDamage,WhiteDropDamage,BlurDamage; Distortion: distortion
- YAML-driven pipelines via `load_pipeline_from_yaml`
- Easy PyTorch integration
  - Single-pass augmentation (`AugupipeWrapper`)
  - Multi-augmentation dataset (`MultiAugDataset`)

---

## Structure

```
augpipe/
├── setup.py
├── requirements.txt
├── pipeline_loader.py      # YAML → ComposeTransform
├── wrappers.py             # AugpipeWrapper, ToTensorList (optional)
├── transforms/
│   ├── base.py
│   ├── geometric.py
│   ├── damage.py
│   ├── distortion.py
│   └── compose.py
├── configs/
│   ├── all_basic.yaml
│   └── one_of_all.yaml
└── utils.py                # Optional helpers
```

---

## 🧩 Installation

```bash
git clone -b development https://github.com/gavcarp/DLP-2025.git
cd DLP-2025/Task_1/Augmentation/augpipe
pip install -e .
```

---

To test, after installation,
Run 
python demo.py
or batch.py
(in Augmentation/augpipe/scripts_example)

or test_plug.py
or test_preprocess.py
(in Augmentation)
## YAML Configuration Example and explain


```yaml
mode: random
pipeline:
  - Rotate:
      angle: 8
  - Scale:
      min_factor: 0.85
      max_factor: 1.15
  
  - Translate:
      x: 3
      y: 2
  - Distortion:
      amp: 2.5
      sigma: 20.0
  - OneOf:
      - WhiteDropDamage:
          drop_length: 4
          drop_width: 4
      - BlackDropDamage:
          num_drops: 20
          min_len: 2
          max_len: 6
          thickness: 2
  - Sometimes:
      p: 0.5
      transform:
        BlurDamage:
          kernel: 7
          sigma: 2.0

```
# Augupipe YAML Example: Explanation and Usage

This is a sample augmentation pipeline configuration for use with the `augupipe` library. It demonstrates how to compose a sequence of image transformations using YAML in a flexible and human-readable way.

## Mode

The top-level field `mode: random` sets the pipeline to apply **randomized parameters** for each transform on each image. If you prefer to apply fixed parameters (for testing or reproducibility), you can instead write:

```
mode: fixed
```

In `"fixed"` mode, each transformation will always use the exact values specified in the YAML.

## Transform Pipeline

The `pipeline` key defines an ordered list of transformations. Here's what each component in the example does:

### 1. Rotate

```yaml
- Rotate:
    angle: 8
```

This rotates the image randomly between -8 and +8 degrees when in `"random"` mode. If in `"fixed"` mode, the image will always be rotated by exactly 8 degrees.

### 2. Scale

```yaml
- Scale:
    min_factor: 0.85
    max_factor: 1.15
```

This randomly scales the image size between 85% and 115% of the original. The output image is then padded or cropped to maintain the original dimensions. In `"fixed"` mode, the scale is set to the average of the min and max values (i.e., (0.85 + 1.15) / 2).

### 3. Translate

```yaml
- Translate:
    x: 3
    y: 2
```

Applies a translation (shift) along the x and y axes. In `"random"` mode, the x-shift is a random integer between -3 and +3 pixels, and similarly for y. In `"fixed"` mode, it's exactly (3, 2).

### 4. Distortion

```yaml
- Distortion:
    amp: 2.5
    sigma: 20.0
```

Applies a smooth spatial distortion. This can simulate warping or curvature like on a scanned document. `amp` controls the amplitude of distortion; `sigma` controls the smoothness of the distortion field.

### 5. OneOf

```yaml
- OneOf:
    - WhiteDropDamage:
        drop_length: 4
        drop_width: 4
    - BlackDropDamage:
        num_drops: 20
        min_len: 2
        max_len: 6
        thickness: 2
```

`OneOf` selects **only one** of the listed transforms at random each time it's called. In this case, it will either apply `WhiteDropDamage` (which adds white rectangular blotches) or `BlackDropDamage` (which simulates black lines or smudges), depending on the mode.

### 6. Sometimes

```yaml
- Sometimes:
    p: 0.5
    transform:
      BlurDamage:
          kernel: 7
          sigma: 2.0
```

`Sometimes` applies the given transform with a probability `p`. Here, there’s a 50% chance to apply `BlurDamage`, which applies a Gaussian blur with kernel size 7 and sigma 2.0. If the condition fails (random draw > 0.5), no blur is applied.

## How to Use This File

Save this configuration as a `.yaml` file (e.g., `configs/sometime_blur.yaml`). Then in Python, load and apply the pipeline like so:

```python
from augupipe import load_pipeline_from_yaml

pipeline = load_pipeline_from_yaml("configs/sometime_blur.yaml")
augmented_img = pipeline(input_numpy_array)
```

You can also wrap the pipeline for use in PyTorch datasets using `AugupipeWrapper`.

## Notes

- Transforms are always applied in the order listed.
- You can nest `OneOf` and `Sometimes` as deeply as needed.
- You can extend the pipeline with your own custom transforms if they inherit from `BaseTransform`.
---

##  Basic Usage

### Load and apply pipeline:

```python
from augupipe import load_pipeline_from_yaml

pipeline = load_pipeline_from_yaml("configs/one_of_all.yaml")
augmented = pipeline(input_numpy_image)  # numpy H×W×C → H×W×C
```

---

## 🔁 Single-Pass Wrapper for PyTorch

```python
from augupipe import load_pipeline_from_yaml, AugupipeWrapper
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

pipeline = load_pipeline_from_yaml("configs/one_of_all.yaml")
transform = transforms.Compose([
    AugupipeWrapper(pipeline),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("data/images", transform=transform)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Multi-Augmentation Dataset

### `multi_augment_dataset.py`

```python
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MultiAugDataset(Dataset):
    def __init__(self, image_folder, pipeline, n=5):
        self.paths     = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.png','jpg','jpeg'))
        ]
        self.pipeline  = pipeline
        self.n         = n
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img     = Image.open(self.paths[idx])
        img_np  = np.array(img)
        outs    = []
        for _ in range(self.n):
            out_np  = self.pipeline(img_np)
            out_pil = Image.fromarray(out_np)
            out_t   = self.to_tensor(out_pil)
            outs.append(out_t)
        return torch.stack(outs, dim=0)  # Shape: (N, C, H, W)
```

### Usage

```python
from torch.utils.data import DataLoader
from augpipe import load_pipeline_from_yaml
from multi_augment_dataset import MultiAugDataset

pipeline = load_pipeline_from_yaml("configs/one_of_all.yaml")
dataset  = MultiAugDataset("data/images", pipeline, n=3)
loader   = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    print(batch.shape)  # (4, 3, C, H, W)
    break
```

---

##  Notes

- YAML files can reside anywhere — just pass the correct path to `load_pipeline_from_yaml()`.
- `mode` in YAML selects between `random` and `fixed` behavior.
- For multi-modal tasks (e.g., segmentation), integrate augmentation inside your custom `Dataset`.

# 📚 Augupipe API Reference

This document provides a summary of the core classes and functions provided by the `augupipe` image augmentation library.

---

## 🔧 Core Functions

### `load_pipeline_from_yaml(yaml_path: str) -> Callable`
**Module:** `augpipe.pipeline_loader`

Loads a YAML configuration file and returns a composable augmentation pipeline.

**Parameters:**
- `yaml_path`: Path to the YAML file defining the augmentation sequence.

**Returns:**
- A callable object (usually a `ComposeTransform`) that takes a NumPy image and returns an augmented NumPy image.

---

## 🧩 PyTorch Integration

### `AugupipeWrapper`
**Module:** `augpipe.wrappers`

A wrapper to embed a NumPy-based pipeline into a `torchvision.transforms` pipeline.

**Usage Example:**
```python
from augupipe import load_pipeline_from_yaml, AugupipeWrapper
from torchvision import transforms

# Step 1: Load a pipeline defined in YAML
pipeline = load_pipeline_from_yaml("configs/one_of_all.yaml")

# Step 2: Wrap it for torchvision compatibility
transform = transforms.Compose([
    AugupipeWrapper(pipeline),  # <--- Wraps NumPy input/output logic
    transforms.ToTensor(),      # Converts result to PyTorch tensor
])
```

---

## 📁 Utilities

### `load_image(path: str) -> np.ndarray`
**Module:** `augpipe.utils`

Loads an image (grayscale or color) from a file into a NumPy array.

### `save_image(img: np.ndarray, path: str)`
**Module:** `augpipe.utils`

Saves a NumPy image array (H×W or H×W×C) as a PNG file.

---

## 📦 Dataset Classes

### `MultiAugDataset`
**Location:** `multi_augment_dataset.py`

A PyTorch-compatible dataset that returns `n` augmented versions per image.

**Constructor:**
```python
MultiAugDataset(image_folder: str, pipeline: Callable, n: int = 5)
```

**Returns:**
A tensor of shape `(n, C, H, W)` per sample.

---

## 🧱 Transform Modules

Each transform class inherits from `BaseTransform`. You can extend and combine them using `ComposeTransform`, `OneOfTransform`, and `SometimesTransform`.

### Examples:
- **Geometric:** `Rotate`, `Translate`, `Scale`
- **Damage:** `WhiteDropDamage`, `BlackDropDamage`, `BlurDamage`
- **Distortion:** `Distortion`
- **Composition:** `ComposeTransform`, `OneOfTransform`, `SometimesTransform`

---

## 🔄 Transform Composition

### `ComposeTransform`
Chains multiple transforms into a single pipeline.

### `OneOfTransform`
Randomly chooses one transform from a list.

### `SometimesTransform`
Applies a transform with a given probability.

---

For usage examples, see the YAML section and `demo.py`, `batch.py`, or notebook-based tutorials.