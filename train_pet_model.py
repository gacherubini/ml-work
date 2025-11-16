import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10

DATASET_DIR = r"C:\Users\Administrator\Documents\codigos\Ml\stanford_dogs\Images"

def prepare(ds, shuffle=False):
    # aplica preprocessamento
    ds = ds.map(
        lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    )
    # shuffle somente antes do batch (dataset já vem batched)
    if shuffle:
        ds = ds.shuffle(2000)

    # NÃO APLICAR .batch DE NOVO
    ds = ds.prefetch(AUTOTUNE)
    return ds

def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    embedding = tf.keras.layers.Dense(256, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(0.3)(embedding)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    embed_model = tf.keras.Model(inputs, embedding)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, embed_model

def extract_embeddings(ds, embed_model):
    embeddings, labels, images = [], [], []
    for batch_img, batch_lbl in ds:
        emb = embed_model.predict(batch_img, verbose=0)
        embeddings.append(emb)
        labels.append(batch_lbl.numpy())
        images.append(batch_img.numpy())
    return np.vstack(embeddings), np.concatenate(labels), np.concatenate(images)

def main():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        validation_split=0.2,
        subset="training",
        seed=123
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        validation_split=0.2,
        subset="validation",
        seed=123
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = prepare(train_ds, shuffle=True)
    val_ds = prepare(val_ds)

    model, embed_model = build_model(num_classes)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save("dog_model.keras")
    embed_model.save("dog_embedding_model.keras")

    with open("dog_class_names.txt", "w") as f:
        for c in class_names: 
            f.write(c + "\n")

    print("Extraindo embeddings...")
    emb, lbl, imgs = extract_embeddings(train_ds, embed_model)

    np.save("dog_train_embeddings.npy", emb)
    np.save("dog_train_labels.npy", lbl)
    np.save("dog_train_images.npy", imgs)

    print("Treino completo!")

if __name__ == "__main__":
    main()
