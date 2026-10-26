# ArchiMind-3D: Language-Guided Multi-Object Grounding

Developed by the ArchiMind Team for the 2026 VLP Challenge.

2026 VLP 挑战赛参赛作品

This repository contains the Python source code for the VLMOD candidate-language matching experiments (Baseline-3A and Baseline-3B), plus optional earlier geometric regression baselines when available. The primary model is a PyTorch multilayer perceptron (MLP) operating on structured candidate-object properties and textual-condition features. **It is not an image-based end-to-end vision-language network.**

## Method and limits

- Training labels: supplied training `label_3` records grouped by natural-language descriptions.
- Candidate training pools are reconstructed from the **union of annotated positives** per image, not complete scene candidates. Internal validation F1 is consequently biased and **is not the official VLP score**.
- Testing: the current inference pipeline uses the official test JSON `test_data` as candidate-object input. **Its eligibility under contest rules must be checked by the submitting team.** This code does not contain ground-truth labels for the test set.
- Geometry (`position`, `size`, `rotation`) in the draft output is **copied from the provided candidate record**, not independently regressed by the MLP. The model predicts object-description association confidence.
- The produced JSON is a **proposed submission schema**, not a verified official schema. Adjust the adapter if the online evaluator expects different keys.

## Requirements

Python 3.10+ recommended. Install with:

```bash
python -m pip install -r requirements.txt
```

## Dataset placement (not redistributed)

Download data directly from the competition platform according to its usage restrictions. Place the folders locally at:

```text
VLMOD(train+test)/MonoMulti3D/train/*.json
VLMOD(train+test)/MonoMulti3D/test/*.json
data/official/val.jsonl
```

The train/validation split may use `data/official/val.jsonl` created by the preprocessing script. **Original datasets and annotations are not part of this public repository.**

## Training Baseline-3B

```bash
python scripts/baseline3b_torch.py --mode train --root . --out experiments/Baseline_3B --epochs 15 --batch-size 512
```

## Export ONNX

```bash
python scripts/baseline3b_torch.py --mode export --weights experiments/Baseline_3B/best_model.pth --out experiments/Baseline_3B
```

The ONNX model consumes **normalized feature vectors**, not raw images or JSON strings. Keep its matching preprocessing and normalization with the released weights.

## Predict from candidate test JSON (only where permitted by contest rules)

```bash
python scripts/inference_baseline3b_submission.py --root . --weights experiments/Baseline_3B/best_model.pth --out final_submission/predictions.json --threshold 0.5 --allow-test-candidates
```

`target_id` is the zero-based row index in `test_data`; `query_id` is the zero-based index in `public_description`; `size` is `[width,height,length]`.

## Model weights / reproducibility

Supply a public Baidu Netdisk or Aliyun Drive URL and its extraction code **through the competition submission form**. Share `best_model.pth`, `candidate_matcher.onnx`, and `onnx_normalization.json` (when generated). Do not commit data or weights to this repository. The team must verify that the `.pth` and ONNX outputs agree within numerical tolerance before final submission.

## Results

No official online evaluation score is claimed. Internal validation uses an incomplete candidate pool and cannot be compared directly with geometry-regression baselines. See your own run outputs (`validation_metrics.json` and `threshold_scan.csv`, kept private/local) for reproducible local results.

## License

MIT License (see [LICENSE](LICENSE)). The team must confirm that it owns the rights to all included source files and that third-party dependencies and reused code permit distribution. The competition dataset is separately licensed and **is not covered by this code license**.
