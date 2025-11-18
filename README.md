Dog Breed Classifier + Similarity Search

Este projeto usa TensorFlow + EfficientNet para treinar um classificador de raças de cachorros e, após o treino, permite comparar uma imagem fornecida com as imagens mais semelhantes do dataset usando embeddings.

 1. Baixar o Dataset

Baixe o dataset Stanford Dogs Dataset:

Kaggle:
https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

Depois de extrair, organize a pasta assim:

./stanford_dogs/Images/...
./stanford_dogs/Annotation/...

O código utiliza exatamente este caminho:

DATASET_DIR = "./stanford_dogs/Images"

 2. Instalar Dependências

Recomenda-se um ambiente virtual.

pip install tensorflow pillow matplotlib numpy

 3. Executar o Treino

O script de treino:

Carrega o dataset

Preprocessa as imagens

Treina um modelo com EfficientNetB0

Gera e salva embeddings

Exporta modelos e arquivos auxiliares

Execute:

python train.py


Arquivos gerados:

dog_model.keras
dog_embedding_model.keras
dog_class_names.txt
dog_train_embeddings.npy
dog_train_labels.npy
dog_train_images.npy

 4. Rodar o Demo de Predição + Similaridade

Coloque a imagem que deseja testar, por exemplo:

meu_animal.jpg


Execute:

python demo_similarity.py


O script irá:

Carregar o modelo treinado

Classificar a raça da imagem

Exibir confiança da predição

Calcular o embedding e comparar com o dataset

Mostrar as 5 imagens mais semelhantes em um gráfico

Exemplo de saída:

Raça prevista: Golden_Retriever
Confiança: 0.8721

 Estrutura Final Esperada
.
├── train.py
├── demo_similarity.py
├── stanford_dogs/
│   └── Images/
├── dog_model.keras
├── dog_embedding_model.keras
├── dog_class_names.txt
├── dog_train_embeddings.npy
├── dog_train_labels.npy
├── dog_train_images.npy
└── meu_animal.jpg

 Tecnologias Utilizadas

TensorFlow 2.x

EfficientNetB0

NumPy

Matplotlib

Pillow

Python 3.9+