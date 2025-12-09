"""Utilitaires pour la préparation du dataset vidéo de frottement."""

import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".exr"}


def parse_labels_from_folder_name(folder_name):
    # Extrait les labels matériau et friction à partir du nom de dossier
    parts = folder_name.replace("_", " ").split()
    material_str = None
    friction_str = None

    for token in parts:
        token_lower = token.lower()
        if "mat" in token_lower:
            material_str = token_lower.replace("mat", "")
        if "frict" in token_lower:
            friction_str = token_lower.replace("frict", "")

    if material_str is None or friction_str is None:
        raise ValueError(f"Nom de dossier invalide pour les labels: {folder_name}")

    material = int(material_str)
    friction = float(friction_str)

    return material, friction


def discover_dataset(root_dir, material_classes=8, friction_step=0.1, max_friction=1.5):
    # Parcourt le dataset et construit les listes de chemins et labels
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {root}")

    video_paths = []
    labels_material = []
    labels_friction = []

    num_bins = int(max_friction / friction_step) + 1

    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        try:
            material, friction = parse_labels_from_folder_name(entry.name)
        except Exception:
            continue

        material_index = material - 1
        if material_index < 0 or material_index >= material_classes:
            continue

        friction_index = int(round(friction / friction_step))
        if friction_index < 0 or friction_index >= num_bins:
            continue

        has_video_or_frames = False
        for child in entry.iterdir():
            if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                has_video_or_frames = True
                break
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                has_video_or_frames = True
                break
        if not has_video_or_frames:
            continue

        video_paths.append(str(entry))
        labels_material.append(material_index)
        labels_friction.append(friction_index)

    if not video_paths:
        raise RuntimeError("Aucune séquence valide détectée dans le dataset.")

    return (
        np.array(video_paths),
        np.array(labels_material, dtype=np.int32),
        np.array(labels_friction, dtype=np.int32),
        num_bins,
    )


def split_dataset(video_paths, y_material, y_friction, val_size=0.15, test_size=0.15, random_state=42):
    # Découpe le dataset en train/val/test
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size doit être < 1.0")

    temp_size = val_size + test_size
    x_train, x_temp, y_train_mat, y_temp_mat, y_train_fric, y_temp_fric = train_test_split(
        video_paths,
        y_material,
        y_friction,
        test_size=temp_size,
        random_state=random_state,
        stratify=y_material,
    )

    relative_val_size = val_size / temp_size
    x_val, x_test, y_val_mat, y_test_mat, y_val_fric, y_test_fric = train_test_split(
        x_temp,
        y_temp_mat,
        y_temp_fric,
        test_size=1.0 - relative_val_size,
        random_state=random_state,
        stratify=y_temp_mat,
    )

    return (
        x_train,
        y_train_mat,
        y_train_fric,
        x_val,
        y_val_mat,
        y_val_fric,
        x_test,
        y_test_mat,
        y_test_fric,
    )


def load_video_frames(path, timesteps, target_size):
    # Charge un nombre fixe d'images à partir d'une vidéo ou d'un dossier
    path_obj = Path(path)

    if path_obj.is_file() and path_obj.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(path_obj))
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise RuntimeError("Nombre de frames vidéo invalide")

            indices = np.linspace(0, frame_count - 1, timesteps).astype(int)
            frames = []
            last_valid = None

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    if last_valid is None:
                        continue
                    frames.append(last_valid.copy())
                    continue
                last_valid = frame
                frames.append(frame)

        finally:
            cap.release()

    else:
        if not path_obj.is_dir():
            raise FileNotFoundError(f"Chemin vidéo invalide: {path_obj}")

        images = sorted(
            [
                p
                for p in path_obj.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        if not images:
            raise RuntimeError(f"Aucune image trouvée dans {path_obj}")

        indices = np.linspace(0, len(images) - 1, timesteps).astype(int)
        frames = []
        last_valid = None

        for idx in indices:
            img_path = images[idx]
            frame = cv2.imread(str(img_path))
            if frame is None:
                if last_valid is None:
                    continue
                frames.append(last_valid.copy())
                continue
            last_valid = frame
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"Aucun frame valide pour {path_obj}")

    if len(frames) < timesteps:
        last = frames[-1]
        while len(frames) < timesteps:
            frames.append(last.copy())

    processed = []
    target_w, target_h = target_size
    for frame in frames:
        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed.append(rgb)

    arr = np.stack(processed, axis=0).astype("float32") / 255.0
    return arr


class VideoSequence(Sequence):
    # Générateur Keras pour le modèle de base matériau
    def __init__(self, video_paths, labels, batch_size, timesteps=16, target_size=(128, 128), shuffle=True):
        self.video_paths = np.array(video_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.video_paths))
        self.on_epoch_end()

    def __len__(self):
        # Nombre de batches par époque
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        # Construit un batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.video_paths[batch_indices]
        batch_labels = self.labels[batch_indices]

        x_batch = []
        for path in batch_paths:
            try:
                frames = load_video_frames(path, self.timesteps, self.target_size)
            except Exception:
                frames = np.zeros((self.timesteps, self.target_size[1], self.target_size[0], 3), dtype="float32")
            x_batch.append(frames)

        x_batch = np.stack(x_batch, axis=0)
        return x_batch, batch_labels

    def on_epoch_end(self):
        # Mélange des indices entre les époques
        if self.shuffle:
            np.random.shuffle(self.indices)


class ConditionalVideoSequence(Sequence):
    # Générateur Keras pour le modèle conditionnel sur les matériaux
    def __init__(self, video_paths, material_probas, labels_friction, batch_size, timesteps=16, target_size=(128, 128), shuffle=True):
        self.video_paths = np.array(video_paths)
        self.material_probas = np.array(material_probas)
        self.labels_friction = np.array(labels_friction)
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.video_paths))
        self.on_epoch_end()

    def __len__(self):
        # Nombre de batches par époque
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        # Construit un batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.video_paths[batch_indices]
        batch_mat = self.material_probas[batch_indices]
        batch_labels = self.labels_friction[batch_indices]

        x_batch = []
        for path in batch_paths:
            try:
                frames = load_video_frames(path, self.timesteps, self.target_size)
            except Exception:
                frames = np.zeros((self.timesteps, self.target_size[1], self.target_size[0], 3), dtype="float32")
            x_batch.append(frames)

        x_batch = np.stack(x_batch, axis=0)
        return [x_batch, batch_mat], batch_labels

    def on_epoch_end(self):
        # Mélange des indices entre les époques
        if self.shuffle:
            np.random.shuffle(self.indices)
