"""Outil pour tracer les courbes d'entraînement Keras à partir d'un CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    # Parse les arguments de ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_csv", type=str, required=True, help="Fichier CSV généré par CSVLogger")
    parser.add_argument("--output_path", type=str, default="training_curves.png")
    return parser.parse_args()


def main():
    # Charge un CSV Keras et trace loss/accuracy
    args = parse_args()
    history_path = Path(args.history_csv)
    output_path = Path(args.output_path)

    df = pd.read_csv(history_path)

    plt.figure()
    if "loss" in df.columns:
        plt.plot(df["epoch"], df["loss"], label="loss")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    if "accuracy" in df.columns:
        plt.plot(df["epoch"], df["accuracy"], label="accuracy")
    if "val_accuracy" in df.columns:
        plt.plot(df["epoch"], df["val_accuracy"], label="val_accuracy")

    plt.xlabel("epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    main()
