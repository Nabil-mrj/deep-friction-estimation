"""Évaluation du modèle de frottement conditionnel."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from dataset_utils import discover_dataset, VideoSequence, ConditionalVideoSequence


def parse_args():
    # Parse les arguments de ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Racine du dataset")
    parser.add_argument("--material_model_path", type=str, required=True, help="Chemin du modèle matériau")
    parser.add_argument("--friction_model_path", type=str, required=True, help="Chemin du modèle frottement")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="eval_friction")
    parser.add_argument("--friction_step", type=float, default=0.1)
    return parser.parse_args()


def compute_material_probas(video_paths, labels_dummy, material_model, timesteps, width, height, batch_size):
    # Calcule les probabilités de matériau pour chaque séquence
    seq = VideoSequence(
        video_paths,
        labels_dummy,
        batch_size=batch_size,
        timesteps=timesteps,
        target_size=(width, height),
        shuffle=False,
    )
    probas = material_model.predict(seq, verbose=0)
    return probas


def main():
    # Évalue le modèle de frottement sur l'ensemble du dataset
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths, y_material, y_friction, num_friction_bins = discover_dataset(str(data_root))

    y_fric_cat = tf.keras.utils.to_categorical(y_friction, num_classes=num_friction_bins)

    material_model = tf.keras.models.load_model(args.material_model_path)
    friction_model = tf.keras.models.load_model(args.friction_model_path)

    dummy = np.zeros_like(y_material)

    material_probas = compute_material_probas(
        video_paths,
        dummy,
        material_model,
        timesteps=args.timesteps,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )

    seq = ConditionalVideoSequence(
        video_paths,
        material_probas,
        y_fric_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=False,
    )

    metrics = friction_model.evaluate(seq, verbose=0)
    eval_path = output_dir / "friction_eval_metrics.txt"
    with eval_path.open("w") as f:
        for name, value in zip(friction_model.metrics_names, metrics):
            f.write(f"{name}: {value}\n")

    y_pred_probas = friction_model.predict(seq, verbose=0)
    y_pred_bins = np.argmax(y_pred_probas, axis=1)
    y_true_bins = np.argmax(y_fric_cat, axis=1)

    report = classification_report(y_true_bins, y_pred_bins, digits=4)
    report_path = output_dir / "friction_classification_report.txt"
    with report_path.open("w") as f:
        f.write(report)

    y_true_fric = y_true_bins * args.friction_step
    y_pred_fric = y_pred_bins * args.friction_step
    rmse = float(np.sqrt(np.mean((y_true_fric - y_pred_fric) ** 2)))

    rmse_path = output_dir / "friction_rmse.txt"
    with rmse_path.open("w") as f:
        f.write(f"RMSE_friction: {rmse}\n")


if __name__ == "__main__":
    main()
