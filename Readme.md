# Emotion Detector (4 Classes)

**Classes:** `angry`, `happy`, `sad`, `surprised`

A Keras/TensorFlow notebook project that trains an image-based emotion detector (4 classes) using modern CNN backbones (EfficientNetB0 used in the notebook). The notebook includes data loading, preprocessing, model building, training, evaluation (accuracy + confusion matrix), and saving/loading model weights.

---

## Project overview

This repository contains a Jupyter notebook (`Emotion_Detector_4_Class.ipynb`) that implements an emotion recognition pipeline:

- dataset reading and counting (`count_images` helper present)
- preprocessing utility (`preprocess_input`, `load_and_prep`)
- model construction using a Keras application backbone (EfficientNetB0 referenced)
- training loop (with `epochs`, `batch_size`, `learning_rate`, `optimizer` parameters)
- evaluation (accuracy, confusion matrix via `sklearn.metrics`)
- saving and loading trained models (examples: `emotion_effnetb0.h5`, `emotion_model.h5`, `emotion_model.keras`)

The notebook aims for ≥ 90% accuracy on the user's test set (this is the stated goal inside the notebook).

---

## Requirements

Recommended environment:

- Python 3.8 or newer
- TensorFlow 2.x
- numpy
- scikit-learn
- pillow (PIL)
- matplotlib (for plots)
- jupyter or jupyterlab

Install with pip:

```bash
pip install tensorflow numpy scikit-learn pillow matplotlib jupyterlab
```

> Note: If you plan to use GPU acceleration, install the GPU build of TensorFlow appropriate for your CUDA/cuDNN versions.

---

## Files (from the notebook environment)

- `Emotion_Detector_4_Class.ipynb` — main notebook containing code and instructions.
- `emotion_effnetb0.h5` — example saved model filename referenced in the notebook.
- `emotion_model.h5` / `emotion_model.keras` — other example saved model filenames.
- `./data/` or `./dataset/` (expected dataset directories) — the notebook expects dataset folders such as `./data/train`, `./data/val`, `./data/test` or similar.

---

## Dataset structure (expected)

The notebook assumes a directory-per-class layout commonly used with Keras `image_dataset_from_directory` or `flow_from_directory`:

```
dataset/
  train/
    angry/
    happy/
    sad/
    surprised/
  val/
    angry/
    happy/
    sad/
    surprised/
  test/
    angry/
    happy/
    sad/
    surprised/
```

If you're on Windows and using absolute paths, use raw strings, e.g. `r"C:\Users\you\Downloads\ML_Projects\Emotion Detector"`.

---

## How to run (notebook)

1. Open the notebook in Jupyter or JupyterLab:

```bash
jupyter lab Emotion_Detector_4_Class.ipynb
# or
jupyter notebook Emotion_Detector_4_Class.ipynb
```

2. Run cells **top → bottom**, one at a time, as indicated in the notebook.

3. Edit the dataset path cell to point to your dataset folder if necessary.

4. Configure training hyperparameters (e.g. `epochs`, `batch_size`, `learning_rate`, `optimizer`) in the designated cell before launching training.

---

## Typical usage snippets (from notebook patterns)

### Loading data (example pattern)

```python
from tensorflow.keras.preprocessing import image_dataset_from_directory
train_ds = image_dataset_from_directory(
    'dataset/train',
    image_size=(224,224),
    batch_size=32,
    label_mode='categorical'
)
```

