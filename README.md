# ArchiMind-3D: Language-Guided Multi-Object Grounding

**ArchiMind Team | 2026 VLP Challenge | VLMOD Track**

2026 VLP 挑战赛参赛作品

## 1. Overview

ArchiMind-3D is a candidate-aware language-guided multi-object grounding framework developed for the VLMOD (Visual-Language Multi-Object Detection) task.

The framework associates natural-language descriptions with multiple candidate objects in a three-dimensional scene. It uses structured object attributes and language-derived features to estimate object-description matching probabilities and identify relevant targets.

The repository includes two candidate-matching baselines:

- **Baseline-3A:** A machine-learning-based candidate classification model.
- **Baseline-3B:** A PyTorch multilayer perceptron (MLP) for candidate-description association.

Additional scripts support dataset inspection, preprocessing, validation, and comparison with earlier geometric regression baselines.

The current Baseline-3B model performs candidate-level association using structured features. It does not directly process raw image pixels or independently estimate three-dimensional object geometry from monocular images.

## 2. Methodology

### 2.1 Candidate object representation

Each candidate object is represented by structured attributes extracted from the provided annotation or candidate records. These attributes describe object properties such as category, appearance, spatial position, dimensions, and orientation.

Natural-language descriptions provide the conditions used to identify relevant objects within a candidate set.

### 2.2 Candidate-description matching

For each natural-language description, object attributes and language-derived features are combined into a candidate-level feature representation.

Baseline-3B uses a multilayer perceptron to estimate the probability that a candidate object matches the description.

A confidence threshold is subsequently applied to determine the selected targets. Multiple objects may be selected for a single description.

### 2.3 Prediction outputs

The inference pipeline generates JSON predictions containing:

- `image_id`: Identifier of the input scene or image.
- `query_id`: Zero-based index of the natural-language description.
- `target_id`: Zero-based index of the selected object in the candidate list.
- `position`: Three-dimensional object coordinates `[x, y, z]`.
- `size`: Object dimensions `[width, height, length]`.
- `rotation`: Object orientation.
- `confidence`: Predicted candidate-description matching probability.

The geometry fields are obtained from the supplied candidate records. The Baseline-3B model predicts association confidence rather than independently regressing object position, dimensions, or orientation.

## 3. Repository Structure

```text
ArchiMind-3D/
├── models/
│   └── baseline.py
├── scripts/
│   ├── baseline3_candidate_model.py
│   ├── baseline3_candidate_probe.py
│   ├── baseline3b_torch.py
│   ├── inference_baseline3b_submission.py
│   ├── compare_baselines_strict.py
│   ├── evaluate_distance_f1.py
│   ├── inspect_official_data.py
│   ├── predict_and_tune_baseline2.py
│   ├── prepare_official_data.py
│   └── tune_confidence_threshold.py
├── utils/
│   └── dataset.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

The `scripts` directory contains the main training, inference, preprocessing, and evaluation utilities. The `models` and `utils` directories contain supporting components for the earlier experimental baselines.

## 4. Installation

Python 3.10 or later is recommended.

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/zhaowillway/ArchiMind-3D-Object-Grounding.git
cd ArchiMind-3D-Object-Grounding
python -m pip install -r requirements.txt
```

The training and inference scripts are designed to run locally with PyTorch and the required Python dependencies.

## 5. Dataset Preparation

The experiments use the VLMOD dataset distributed through the official competition platform.

**Dataset access:** http://vlp.chd.edu.cn/

After obtaining the dataset, organize the annotation files under the project root:

```text
VLMOD(train+test)/
└── MonoMulti3D/
    ├── train/
    │   └── *.json
    └── test/
        └── *.json
```

The training annotations contain natural-language descriptions and corresponding target-object labels. Test annotations provide natural-language descriptions and candidate-object records.

The preprocessing utilities support dataset inspection and preparation of local training and validation files. Depending on the preprocessing workflow, a validation manifest may be stored at:

```text
data/official/val.jsonl
```

Original images, annotations, and other official dataset files are not redistributed through this repository.

The competition dataset is subject to its own licensing and usage restrictions.

## 6. Training

### 6.1 Baseline-3A

Baseline-3A implements candidate-level classification using structured object and language-derived features.

```bash
python scripts/baseline3_candidate_model.py --mode train --root . --out experiments/Baseline_3 --iterations 120
```

Training outputs include model parameters, validation metrics, and confidence-threshold evaluation results.

### 6.2 Baseline-3B

Baseline-3B implements candidate-description matching using a PyTorch multilayer perceptron.

```bash
python scripts/baseline3b_torch.py \
    --mode train \
    --root . \
    --out experiments/Baseline_3B \
    --epochs 15 \
    --batch-size 512
```

The trained model is saved as:

```text
experiments/Baseline_3B/best_model.pth
```

Additional training outputs include validation metrics, threshold scans, and training history.

## 7. ONNX Export

The trained PyTorch model can be exported to ONNX for model deployment and reproducibility.

```bash
python scripts/baseline3b_torch.py \
    --mode export \
    --weights experiments/Baseline_3B/best_model.pth \
    --out experiments/Baseline_3B
```

The exported files include:

```text
experiments/Baseline_3B/
├── best_model.pth
├── candidate_matcher.onnx
└── onnx_normalization.json
```

The ONNX model receives normalized candidate-level feature vectors. It does not directly accept raw images or natural-language strings.

The corresponding feature extraction and normalization procedures are therefore part of the complete inference pipeline.

## 8. Inference

The inference pipeline processes test JSON files containing natural-language descriptions and candidate-object records.

```bash
python scripts/inference_baseline3b_submission.py \
    --root . \
    --weights experiments/Baseline_3B/best_model.pth \
    --out final_submission/predictions.json \
    --threshold 0.5 \
    --allow-test-candidates
```

The `--allow-test-candidates` flag enables inference using the candidate objects provided in the test JSON files.

The output is written to:

```text
final_submission/predictions.json
```

### Output format

An illustrative prediction structure is shown below. The numerical values are examples rather than actual model outputs.

```json
[
  {
    "image_id": "example_image",
    "predictions": [
      {
        "query_id": 0,
        "objects": [
          {
            "target_id": 2,
            "position": [1.2, 0.5, 25.6],
            "size": [1.8, 1.5, 4.3],
            "rotation": 1.57,
            "confidence": 0.95
          }
        ]
      }
    ]
  }
]
```

The output format represents associations between natural-language queries and candidate objects. The field organization follows the repository's inference implementation.

## 9. Experimental Evaluation

### 9.1 Internal validation

The candidate-matching models are evaluated using precision, recall, and F1 score.

Training and validation candidate pools are reconstructed from the union of annotated positive objects within each scene. These reconstructed pools do not necessarily contain all objects appearing in the scene.

Consequently, the internal validation results characterize matching performance within the reconstructed candidate sets and may overestimate performance on complete candidate pools.

### 9.2 Baseline comparison

The repository includes scripts for evaluating earlier geometric regression baselines and candidate-based matching methods.

The earlier regression models and the candidate-classification models use different prediction formulations and evaluation conditions. Their numerical results are therefore not directly interchangeable.

### 9.3 Evaluation status

The internal validation metrics are distinct from the official VLP benchmark scores.

The primary experimental outputs include:

```text
validation_metrics.json
threshold_scan.csv
history.json
```

These files are generated during local model training and evaluation.

## 10. Model Weights and Reproducibility

The Baseline-3B model supports PyTorch and ONNX formats.

The principal model artifacts are:

| File | Description |
|---|---|
| `best_model.pth` | Trained PyTorch model weights |
| `candidate_matcher.onnx` | Exported ONNX model |
| `onnx_normalization.json` | Feature normalization parameters required for ONNX inference |

The model can be retrained using the scripts and data preparation procedures described in this repository.

Model weight files and original competition datasets are distributed separately from the source-code repository.

### Pre-trained Model Weights

The trained Baseline-3B model weights and ONNX files are available from Baidu Netdisk:

- **Download:** [Baidu Netdisk – ArchiMind-3D Model Weights](https://pan.baidu.com/s/10jbdyTipyYWHj5JgMK_2lQ?pwd=u2pi)
- **Extraction code:** `u2pi`

The model package includes the PyTorch weights, ONNX model, and feature normalization configuration.

## 11. License

This repository is released under the **MIT License**. See [LICENSE](LICENSE) for the complete license terms.

Copyright (c) 2026 ArchiMind Team.

The official VLMOD dataset is separately licensed and is not covered by the MIT License of this repository.
