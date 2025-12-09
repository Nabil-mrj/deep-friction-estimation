"""Script d'entraînement du modèle conditionnel pour le frottement."""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_utils import (
    discover_dataset,
    split_dataset,
    VideoSequence,
    ConditionalVideoSequence,
)
from models import create_conditional_friction_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    # Parse les arguments de ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Racine du dataset")
    parser.add_argument("--material_model_path", type=str, required=True, help="Modèle matériau entraîné")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="outputs_friction")
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
    probas = material_model.predict(seq)
    return probas


def main():
    # Entraîne le modèle conditionnel de frottement
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        video_paths,
        y_material,
        y_friction,
        num_friction_bins,
    ) = discover_dataset(str(data_root))

    num_materials = int(np.max(y_material)) + 1

    (
        x_train,
        y_train_mat,
        y_train_fric,
        x_val,
        y_val_mat,
        y_val_fric,
        x_test,
        y_test_mat,
        y_test_fric,
    ) = split_dataset(
        video_paths,
        y_material,
        y_friction,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    y_train_fric_cat = tf.keras.utils.to_categorical(y_train_fric, num_classes=num_friction_bins)
    y_val_fric_cat = tf.keras.utils.to_categorical(y_val_fric, num_classes=num_friction_bins)
    y_test_fric_cat = tf.keras.utils.to_categorical(y_test_fric, num_classes=num_friction_bins)

    input_shape = (args.timesteps, args.height, args.width, 3)

    material_model = tf.keras.models.load_model(args.material_model_path)

    dummy_train = np.zeros_like(y_train_mat)
    dummy_val = np.zeros_like(y_val_mat)
    dummy_test = np.zeros_like(y_test_mat)

    material_probas_train = compute_material_probas(
        x_train,
        dummy_train,
        material_model,
        timesteps=args.timesteps,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )
    material_probas_val = compute_material_probas(
        x_val,
        dummy_val,
        material_model,
        timesteps=args.timesteps,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )
    material_probas_test = compute_material_probas(
        x_test,
        dummy_test,
        material_model,
        timesteps=args.timesteps,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )

    train_seq = ConditionalVideoSequence(
        x_train,
        material_probas_train,
        y_train_fric_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=True,
    )
    val_seq = ConditionalVideoSequence(
        x_val,
        material_probas_val,
        y_val_fric_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=False,
    )
    test_seq = ConditionalVideoSequence(
        x_test,
        material_probas_test,
        y_test_fric_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=False,
    )

    model = create_conditional_friction_model(input_shape, num_materials, num_friction_bins)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "friction_lrcn_best.h5"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(output_dir / "friction_history.csv"),
            separator=",",
            append=False,
        ),
    ]

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False,
    )

    history_path = output_dir / "friction_history.npy"
    np.save(history_path, history.history, allow_pickle=True)

    test_metrics = model.evaluate(test_seq)
    metrics_path = output_dir / "friction_test_metrics.txt"
    with metrics_path.open("w") as f:
        for name, value in zip(model.metrics_names, test_metrics):
            f.write(f"{name}: {value}\n")

    y_true_bins = np.argmax(y_test_fric_cat, axis=1)
    y_pred_probas = model.predict(test_seq)
    y_pred_bins = np.argmax(y_pred_probas, axis=1)

    y_true_fric = y_true_bins * args.friction_step
    y_pred_fric = y_pred_bins * args.friction_step
    rmse = float(np.sqrt(np.mean((y_true_fric - y_pred_fric) ** 2)))

    rmse_path = output_dir / "friction_rmse.txt"
    with rmse_path.open("w") as f:
        f.write(f"RMSE_friction: {rmse}\n")

    model.save(str(output_dir / "friction_lrcn_final"))


if __name__ == "__main__":
    main()
