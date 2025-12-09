"""Évaluation du modèle de classification de matériau."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset_utils import discover_dataset, VideoSequence


def parse_args():
    # Parse les arguments de ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Racine du dataset")
    parser.add_argument("--material_model_path", type=str, required=True, help="Chemin du modèle matériau")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="eval_material")
    return parser.parse_args()


def main():
    # Évalue le modèle matériau sur l'ensemble du dataset
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths, y_material, y_friction, num_friction_bins = discover_dataset(str(data_root))
    num_materials = int(np.max(y_material)) + 1

    y_material_cat = tf.keras.utils.to_categorical(y_material, num_classes=num_materials)

    seq = VideoSequence(
        video_paths,
        y_material_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=False,
    )

    model = tf.keras.models.load_model(args.material_model_path)

    metrics = model.evaluate(seq, verbose=0)
    eval_path = output_dir / "material_eval_metrics.txt"
    with eval_path.open("w") as f:
        for name, value in zip(model.metrics_names, metrics):
            f.write(f"{name}: {value}\n")

    y_pred_probas = model.predict(seq, verbose=0)
    y_pred = np.argmax(y_pred_probas, axis=1)
    y_true = y_material

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    report_path = output_dir / "material_classification_report.txt"
    with report_path.open("w") as f:
        f.write(report)
        f.write("\n\nConfusion matrix:\n")
        f.write(str(cm))


if __name__ == "__main__":
    main()
