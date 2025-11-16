import os

# ----- SILENCIAR TUDO DO TENSORFLOW -----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import contextlib
import io

null_stream = io.StringIO()
@contextlib.contextmanager
def silence():
    with contextlib.redirect_stdout(null_stream):
        with contextlib.redirect_stderr(null_stream):
            yield

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = (224, 224)

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.array(img)

def preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)

def main():
    print("Carregando modelos...")

    with silence():
        model = tf.keras.models.load_model("dog_model.keras", compile=False)
    with silence():
        embed_model = tf.keras.models.load_model("dog_embedding_model.keras", compile=False)

    print("Modelos carregados!")

    print("Carregando classes...")
    class_names = [x.strip() for x in open("dog_class_names.txt").readlines()]

    # SUA IMAGEM AQUI
    image_path = "./meu_animal4.jpg"

    img = load_image(image_path)
    proc = np.expand_dims(preprocess(img.copy()), axis=0)

    preds = model.predict(proc, verbose=0)[0]
    top = np.argmax(preds)
    conf = preds[top]

    print("\n=====================")
    print(f"Raça prevista: {class_names[top]}")
    print(f"Confiança: {conf:.4f}")
    print("=====================\n")

    print("Carregando embeddings...")
    train_emb = np.load("dog_train_embeddings.npy")
    train_lbl = np.load("dog_train_labels.npy")
    train_imgs = np.load("dog_train_images.npy")

    print("Criando embedding da imagem...")
    emb = embed_model.predict(proc, verbose=0)[0]

    dists = np.linalg.norm(train_emb - emb, axis=1)
    idx = np.argsort(dists)[:5]

    print("Imagens mais semelhantes:")

    fig, axes = plt.subplots(1, 6, figsize=(18, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Input\n({class_names[top]})")
    axes[0].axis("off")

    for i in range(5):
        axes[i + 1].imshow(train_imgs[idx[i]].astype("uint8"))
        axes[i + 1].set_title(class_names[train_lbl[idx[i]]])
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
