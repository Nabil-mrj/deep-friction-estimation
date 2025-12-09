"""Définition des architectures pour l'estimation matériau et frottement."""

import os

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    LSTM,
    Dropout,
    TimeDistributed,
    GlobalAveragePooling2D,
    Concatenate,
)
from tensorflow.keras.models import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def build_video_encoder(input_shape):
    # Construit un encodeur vidéo partagé (CNN + pooling temporel)
    inputs = Input(shape=input_shape, name="video_input")
    x = TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"))(inputs)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

    x = TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.5)(x)

    return inputs, x


def create_material_model(input_shape, num_materials):
    # Modèle vidéo pour la classification de matériau
    inputs, encoded = build_video_encoder(input_shape)
    x = Dense(128, activation="relu")(encoded)
    x = Dropout(0.5)(x)
    outputs = Dense(num_materials, activation="softmax", name="material_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="material_lrcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_conditional_friction_model(input_shape, num_materials, num_friction_bins):
    # Modèle vidéo conditionnel sur les matériaux pour la discrétisation du frottement
    video_inputs, encoded = build_video_encoder(input_shape)
    material_input = Input(shape=(num_materials,), name="material_probas_input")

    x = Concatenate(name="fusion_video_material")([encoded, material_input])
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_friction_bins, activation="softmax", name="friction_output")(x)

    model = Model(inputs=[video_inputs, material_input], outputs=outputs, name="conditional_friction_lrcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
