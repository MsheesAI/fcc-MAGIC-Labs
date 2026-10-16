
# fcc-MAGIC-Labs

Classifying **gamma** vs. **hadron** particles from the 2004 MAGIC gamma telescope dataset, using classic machine learning models and a small neural network in Python.

The project walks through a full binary classification workflow: loading and exploring the data, visualizing features, splitting and scaling, balancing classes, training five different models, and comparing them with precision, recall, and F1.

## Dataset

`magic04.data` is the MAGIC Gamma Telescope dataset (UCI Machine Learning Repository). It contains simulated Monte Carlo data from a Cherenkov telescope, where each row describes the image a particle shower leaves in the detector.

- **Rows:** 19,020
- **Features:** 10 numeric image parameters
- **Target:** `class`, either `g` (gamma, signal) or `h` (hadron, background)

| Feature | Description |
| --- | --- |
| `flength` | Major axis of the ellipse (mm) |
| `fWidth` | Minor axis of the ellipse (mm) |
| `fSize` | 10-log of the sum of all pixel contents |
| `fConc` | Ratio of the two highest pixels to `fSize` |
| `fConc1` | Ratio of the highest pixel to `fSize` |
| `fAsym` | Distance from the highest pixel to the center, along the major axis (mm) |
| `fM3Long` | Third root of the third moment along the major axis (mm) |
| `fM3Trans` | Third root of the third moment along the minor axis (mm) |
| `fAlpha` | Angle of the major axis with the vector to the origin (deg) |
| `fDist` | Distance from the origin to the ellipse center (mm) |

The data file has no header row, so column names are assigned in the notebook.

## Workflow

The notebook `fcc_MAGIC_Labs.ipynb` covers:

1. **Load and inspect.** Read `magic04.data`, name the columns, and encode the label (`g` = 1, `h` = 0).
2. **Visualize.** Overlaid, density-normalized histograms of every feature for gamma vs. hadron.
3. **Split.** Shuffle and split 60% train / 20% validation / 20% test (11,412 / 3,804 / 3,804 rows).
4. **Preprocess.** Standardize features with `StandardScaler`. Because the training set is imbalanced (7,382 gamma vs. 4,030 hadron), it is balanced with `RandomOverSampler`. Validation and test sets are left untouched.
5. **Model and evaluate.** Train each model on the balanced training set and report a `classification_report` on the test set.

## Models and results

All numbers are on the held-out test set (3,804 samples: 1,324 hadron, 2,480 gamma).

| Model | Accuracy | Hadron F1 | Gamma F1 |
| --- | --- | --- | --- |
| Gaussian Naive Bayes | 0.73 | 0.51 | 0.81 |
| Logistic Regression | 0.79 | 0.71 | 0.83 |
| k-Nearest Neighbors (k=5) | 0.82 | 0.75 | 0.87 |
| Support Vector Machine (SVC, default) | 0.86 | 0.80 | 0.90 |
| Neural Network (TensorFlow/Keras) | **0.88** | **0.82** | **0.91** |

**Neural network configuration:** two hidden `Dense` layers of 64 units with ReLU, each followed by `Dropout(0.2)`, and a sigmoid output. It is compiled with Adam (learning rate 0.001) and binary cross-entropy, and trained for 100 epochs with batch size 128 and a 20% validation split. A helper plots loss and accuracy curves per epoch.

The neural network performs best overall, followed by the SVM. Naive Bayes struggles most with hadron recall (0.41), likely because its feature-independence assumption does not hold for this data.

## Getting started

### Requirements

- Python 3.9+
- numpy, pandas, matplotlib
- scikit-learn
- imbalanced-learn
- tensorflow

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow jupyter
```

### Run it

```bash
git clone https://github.com/MsheesAI/fcc-MAGIC-Labs.git
cd fcc-MAGIC-Labs
jupyter notebook fcc_MAGIC_Labs.ipynb
```

The notebook was authored in Google Colab, so it can also be opened there directly. Make sure `magic04.data` sits in the same directory as the notebook (or upload it to the Colab session), since it is loaded with a relative path.

## Repository structure

```
fcc-MAGIC-Labs/
├── README.md
├── fcc_MAGIC_Labs.ipynb   # Full analysis and model training
└── magic04.data           # MAGIC gamma telescope dataset (19,020 rows)
```

## Notes

- The train/validation/test shuffle is not seeded, so exact metrics will vary slightly between runs.
- Each split is scaled with its own `StandardScaler`. For a stricter setup, fit the scaler on the training data only and reuse it on validation and test.
- The validation set is prepared but not used for tuning in the current notebook, so it is a natural place to extend the project (for example, a hyperparameter search for the neural network).

## Credits

- Dataset: MAGIC Gamma Telescope, UCI Machine Learning Repository.
- Maintained by [MsheesAI](https://github.com/MsheesAI).
