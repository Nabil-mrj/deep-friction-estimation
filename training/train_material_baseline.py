"""Script d'entraînement du modèle de classification de matériau."""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_utils import (
    discover_dataset,
    split_dataset,
    VideoSequence,
)
from models import create_material_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    # Parse les arguments de ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Racine du dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="outputs_material")
    return parser.parse_args()


def main():
    # Entraîne le modèle de classification de matériau
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

    y_train_mat_cat = tf.keras.utils.to_categorical(y_train_mat, num_classes=num_materials)
    y_val_mat_cat = tf.keras.utils.to_categorical(y_val_mat, num_classes=num_materials)
    y_test_mat_cat = tf.keras.utils.to_categorical(y_test_mat, num_classes=num_materials)

    input_shape = (args.timesteps, args.height, args.width, 3)

    train_seq = VideoSequence(
        x_train,
        y_train_mat_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=True,
    )
    val_seq = VideoSequence(
        x_val,
        y_val_mat_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=False,
    )
    test_seq = VideoSequence(
        x_test,
        y_test_mat_cat,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
        target_size=(args.width, args.height),
        shuffle=False,
    )

    model = create_material_model(input_shape, num_materials)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "material_lrcn_best.h5"),
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
            filename=str(output_dir / "material_history.csv"),
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

    history_path = output_dir / "material_history.npy"
    np.save(history_path, history.history, allow_pickle=True)

    test_metrics = model.evaluate(test_seq)
    metrics_path = output_dir / "material_test_metrics.txt"
    with metrics_path.open("w") as f:
        for name, value in zip(model.metrics_names, test_metrics):
            f.write(f"{name}: {value}\n")

    model.save(str(output_dir / "material_lrcn_final"))


if __name__ == "__main__":
    main()
