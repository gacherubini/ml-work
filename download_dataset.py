import tensorflow_datasets as tfds

print("Baixando Oxford-IIIT Pet...")
tfds.load("oxford_iiit_pet", split="train", with_info=True)
print("Download completo!")
