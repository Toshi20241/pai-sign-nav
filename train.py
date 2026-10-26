"""Reproduce synthetic-arrow training and independent image evaluations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from vision import CLASS_NAMES, ROOT, SignClassifier, preprocess, render_sign

SPLITS = {
    "train": {"seed": 11001, "n": 900, "profile": "standard"},
    "validation": {"seed": 22001, "n": 150, "profile": "standard"},
    "test_standard": {"seed": 33001, "n": 300, "profile": "standard"},
    "test_stress": {"seed": 44001, "n": 300, "profile": "stress"},
}
TRAINING_EPOCHS = 60


def display_font(size: int, bold: bool = False):
    """Make optional evidence figures portable across Linux, macOS and Colab."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for candidate in [name, f"/usr/share/fonts/truetype/dejavu/{name}",
                      "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def make_dataset(seed: int, n: int, profile: str):
    if n % 3:
        raise ValueError("datasets must contain equally many examples of each class")
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(3), n // 3)
    rng.shuffle(labels)
    features = np.empty((n, 576), dtype=np.float32)
    examples = []
    digest = hashlib.sha256()
    for i, label in enumerate(labels):
        im = render_sign(int(label), rng, profile)
        features[i] = preprocess(im)
        digest.update(np.asarray(im).tobytes())
        digest.update(bytes([int(label)]))
        if len(examples) < 18:
            examples.append((im, int(label)))
    return features, labels, examples, digest.hexdigest()


def write_contact_sheet(examples, predictions, path: Path, title: str):
    width, height = 900, 500
    canvas = Image.new("RGB", (width, height), "#f2f5f8")
    d = ImageDraw.Draw(canvas)
    font = display_font(15)
    bold = display_font(20, bold=True)
    d.text((22, 15), title, fill="#142b42", font=bold)
    for i, ((im, label), pred) in enumerate(zip(examples, predictions)):
        x, y = 23 + (i % 6) * 147, 55 + (i // 6) * 146
        canvas.paste(im, (x + 19, y))
        d.text((x, y + 99), f"true: {CLASS_NAMES[label]}", fill="#233f52", font=font)
        d.text((x, y + 118), f"MLP: {CLASS_NAMES[int(pred)]}",
               fill="#16714c" if label == pred else "#a82735", font=font)
    canvas.save(path)


def write_confusion(confusion, path: Path):
    canvas = Image.new("RGB", (760, 450), "#f2f5f8")
    d = ImageDraw.Draw(canvas)
    font = display_font(21)
    small = display_font(16)
    row_font = display_font(14)
    d.text((25, 16), "Independent test confusion matrices", fill="#142b42", font=font)
    for k, key in enumerate(["test_standard", "test_stress"]):
        left = 25 + k * 375
        d.text((left, 55), key.replace("test_", "").upper(), fill="#142b42", font=font)
        cm = np.array(confusion[key])
        for j, name in enumerate(CLASS_NAMES):
            d.text((left + 77 + j * 90, 99), name, fill="#344e60", font=small)
        for i, name in enumerate(CLASS_NAMES):
            d.text((left, 149 + i * 85), name, fill="#344e60", font=row_font)
            for j in range(3):
                x, y = left + 77 + j * 90, 129 + i * 85
                fraction = cm[i, j] / max(1, cm[i].sum())
                fill = (int(226 - 184 * fraction), int(240 - 117 * fraction), int(244 - 82 * fraction))
                d.rounded_rectangle((x, y, x + 82, y + 76), radius=8, fill=fill)
                d.text((x + 14, y + 25), str(cm[i, j]), fill="#ffffff" if fraction > .5 else "#17384a", font=font)
    d.text((25, 409), "Rows: true class. Columns: predicted class. 100 images per class.", fill="#455d6a", font=small)
    canvas.save(path)


def main():
    started = time.perf_counter()
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results" / "figures").mkdir(exist_ok=True, parents=True)
    datasets = {}
    dataset_info = {}
    for split, cfg in SPLITS.items():
        print(f"Generating {split}: {cfg['n']}", flush=True)
        x, y, examples, digest = make_dataset(**cfg)
        datasets[split] = (x, y, examples)
        dataset_info[split] = {**cfg, "image_and_label_sha256": digest,
                               "class_counts": np.bincount(y, minlength=3).tolist()}

    x_train, y_train, _ = datasets["train"]
    x_val, y_val, _ = datasets["validation"]
    model = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam",
                          alpha=0.001, batch_size=128, learning_rate_init=0.001,
                          random_state=55001, max_iter=1, shuffle=True)
    history = []
    training_start = time.perf_counter()
    with threadpool_limits(limits=1):
        for epoch in range(1, TRAINING_EPOCHS + 1):
            model.partial_fit(x_train, y_train, classes=np.arange(3))
            p = model.predict_proba(x_val)
            val_loss = float(log_loss(y_val, p, labels=[0, 1, 2]))
            entry = {"epoch": epoch, "training_loss": float(model.loss_),
                     "validation_loss": val_loss,
                     "validation_accuracy": float(accuracy_score(y_val, np.argmax(p, axis=1)))}
            history.append(entry)
            if epoch % 10 == 0:
                print(entry, flush=True)
    weights, biases = model.coefs_, model.intercepts_
    payload = {"n_layers": np.array(len(weights)), "class_names": np.array(CLASS_NAMES),
               "feature_size": np.array(24), "training_epochs": np.array(TRAINING_EPOCHS)}
    for i, (w, b) in enumerate(zip(weights, biases)):
        payload[f"weight_{i}"] = w.astype(np.float32)
        payload[f"bias_{i}"] = b.astype(np.float32)
    np.savez_compressed(ROOT / "models" / "sign_mlp.npz", **payload)
    training_seconds = time.perf_counter() - training_start
    classifier = SignClassifier()
    # Compare library inference to NumPy inference from the exported numeric
    # weights. This verifies export consistency without selecting a model.
    with threadpool_limits(limits=1):
        sklearn_probabilities = model.predict_proba(x_val)
        npz_probabilities = classifier.probabilities(x_val)
    maximum_difference = float(np.max(np.abs(sklearn_probabilities - npz_probabilities)))
    labels_equal = bool(np.array_equal(sklearn_probabilities.argmax(axis=1),
                                      npz_probabilities.argmax(axis=1)))
    if maximum_difference > 1e-6 or not labels_equal:
        raise AssertionError("NPZ inference differs from the trained model")
    metrics = {"task": "three_class_synthetic_arrow_recognition", "class_names": CLASS_NAMES,
               "data_source": "procedurally generated RGB sign patches; no real camera images",
               "splits": dataset_info, "model": {"layers": [576, 32, 3], "activation": "relu",
               "optimizer": "Adam", "learning_rate": 0.001, "alpha": 0.001,
               "batch_size": 128, "training_seed": 55001, "epochs_run": len(history),
               "training_rule": "one run of 60 fixed epochs; validation is monitoring only",
               "training_seconds": training_seconds,
               "parameter_count": int(sum(w.size + b.size for w, b in zip(weights, biases)))},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scikit_learn": sklearn.__version__, "pillow": PIL.__version__},
               "render_profiles": {
                   "standard": {"arrow_scale": [0.78, 1.12], "center_shift_px": [-5, 5],
                                "rotation_deg": [-13, 13], "blur_sigma_px": [0, 0.65],
                                "noise_sd": [0, 6], "background_intensity": [220, 252],
                                "arrow_intensity": [15, 71], "occlusion_probability": 0},
                   "stress": {"arrow_scale": [0.52, 0.95], "center_shift_px": [-11, 11],
                              "rotation_deg": [-38, 38], "blur_sigma_px": [0.5, 2.8],
                              "noise_sd": [7, 20], "background_intensity": [155, 245],
                              "arrow_intensity": [50, 124], "occlusion_probability": 0.45,
                              "occluding_strip_height_px": [5, 12]},
               }, "npz_round_trip_check": {
                   "samples": len(y_val), "max_absolute_probability_difference": maximum_difference,
                   "all_predicted_labels_equal": labels_equal, "tolerance": 1e-6,
                   "passed": maximum_difference <= 1e-6 and labels_equal,
               }, "evaluations": {}, "history": history}
    confusions = {}
    with threadpool_limits(limits=1):
        for split in ["validation", "test_standard", "test_stress"]:
            x, y, examples = datasets[split]
            probs = classifier.probabilities(x)
            prediction = np.argmax(probs, axis=1)
            cm = confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist()
            metrics["evaluations"][split] = {
                "n": len(y), "mlp_correct": int(np.sum(prediction == y)),
                "mlp_accuracy": float(np.mean(prediction == y)),
                "mlp_confusion_matrix": cm,
                "mlp_cross_entropy": float(log_loss(y, probs, labels=[0, 1, 2])),
                "mean_max_probability": float(probs.max(axis=1).mean()),
            }
            confusions[split] = cm
            if split.startswith("test_"):
                with (ROOT / "results" / f"{split}_predictions.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["example_index", "true_label", "mlp_label",
                                     "p_left", "p_straight", "p_right"])
                    for i in range(len(y)):
                        writer.writerow([i, CLASS_NAMES[int(y[i])], CLASS_NAMES[int(prediction[i])],
                                         *[f"{v:.9f}" for v in probs[i]]])
                write_contact_sheet(examples, prediction[:len(examples)],
                                    ROOT / "results" / "figures" / f"{split}_examples.png",
                                    f"Synthetic sign examples / {split.replace('test_', '').upper()}")
    write_confusion(confusions, ROOT / "results" / "figures" / "vision_confusion.png")
    metrics["total_seconds"] = time.perf_counter() - started
    (ROOT / "results" / "vision_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["evaluations"], indent=2), flush=True)
    print(f"Saved model and evidence in {ROOT}; total={metrics['total_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
