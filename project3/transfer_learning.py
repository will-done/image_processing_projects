"""
zatürre sınıflandırması için transfer uygulamaıs
zatrre veriseti: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 

transfer learning model: densenet121
"""

# import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator # görüntü verisi ykleme ve d augment için
from tensorflow.keras.applications import DenseNet121 # önceden eğitilmiş modeli 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # katmanlar 
from tensorflow.keras.models import Model # model oluşturma 
from tensorflow.keras.optimizers import Adam # optimizasyon algoritması
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # callbackler

import matplotlib.pyplot as plt # görselleştirme 
import os # dosya işlemleri
import numpy as np # sayısal işlemler 
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay # model değerlendirme 

# load data ,data augmentation preprocessing 
train_datagen = ImageDataGenerator(
    rescale = 1 /255.0, #normalizasyon
    horizontal_flip = True, # yatay çevirme 
    rotation_range = 10, # rastgele döndürme 
    brightness_range = [0.8, 1.2], # parlaklık ayarı 
    validation_split = 0.1 # doğrulama verisi için ayırma
)
test_datagen = ImageDataGenerator(rescale = 1 /255.0) # test verisi için normalizasyon

DATA_DIR = "chest_xray"
IMG_SIZE = (224, 224) #modelin beklediği input boyutu
BATCH_SIZE = 32 
CLASS_MODE = "binary" # ikili sınıflandırma 

train_gen =  train_datagen.flow_from_directory( 
    os.path.join(DATA_DIR,"train"),
    target_size = IMG_SIZE , # görüntüleri img_size boyutuna yeniden boyutlandırma
    batch_size = BATCH_SIZE,  
    class_mode = CLASS_MODE, # ikili sınıflandırma 
    subset = "training", # eğitim verisi 
    shuffle = True # veriyi karıştırma 
)
val_gen =  train_datagen.flow_from_directory(
    os.path.join(DATA_DIR,"train"),
    target_size = IMG_SIZE , # görüntüleri img_size boyutuna yeniden boyutlandırma
    batch_size = BATCH_SIZE,  
    class_mode = CLASS_MODE, # ikili sınıflandırma 
    subset = "validation", # doğrulama verisi 
    shuffle = False # veriyi karıştırma 
)
test_gen =  test_datagen.flow_from_directory( 
    os.path.join(DATA_DIR,"test"),
    target_size = IMG_SIZE , # görüntüleri img_size boyutuna yeniden boyutlandırma
    batch_size = BATCH_SIZE,  
    class_mode = CLASS_MODE, # ikili sınıflandırma 
    shuffle = False # veriyi karıştırma 
) 



#basic visualization
class_names = list(train_gen.class_indices.keys()) #sınıf isimleri ( normal , pneumonia)
images, labels = next(train_gen) # eğitim verisinden bir batch görüntü ve etiket al 

plt.figure(figsize=(10, 4)) 
for i in range(4): 
    ax = plt.subplot(1, 4, i+1) 
    ax.imshow(images[i]) 
    ax.set_title(class_names[int(labels[i])]) 
    ax.axis("off")
plt.tight_layout() 
plt.show()

# transfer learning modelin tnaımlanamsı: densenet121
base_model = DenseNet121(
    weights = "imagenet", # önceden eğitilmiş ağırlıklar
    include_top = False, # üst katmanları dahil etme
    input_shape = (*IMG_SIZE, 3) # giriş boyutu
) 
base_model.trainable = False # temel modelin ağırlıklarını dondur 

x = base_model.output # temel modelin çıktısı 
x = GlobalAveragePooling2D()(x) # küresel ortalama havuzlama katmanı 
x = Dense(128, activation = "relu")(x) # yoğun katman 
x= Dropout(0.5)(x) # dropout katmanı 
pred = Dense(1, activation = "sigmoid")(x) # çıktı katmanı (ikili sınıflandırma için sigmoid aktivasyonu )

model = Model(inputs = base_model.input, outputs = pred) # modelin tanımlanması

# modelin derlenmesi ve callback
model.compile(
    optimizer = Adam(learning_rate = 1e-4), #optimizer
    loss = "binary_crossentropy", #ikili sınıflandırma kaybı
    metrics = ["accuracy"] #değerlendirme metriği 
)
callbacks = [
    EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True), # erken durdurma 
    ReduceLROnPlateau(monitor = "val_loss", factor = 0.2, patience = 2, min_lr = 1e-6), # öğrenme oranını azaltma 
    ModelCheckpoint("best_model.h5", monitor = "val_loss", save_best_only = True) # en iyi modeli kaydetme 
]
print("model summary:")
print(model.summary())


# modelin eğitilmesi ve sonuçların değerlendirmesi
history = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs = 2, 
    callbacks = callbacks ,
    verbose = 1 # eğitim ilerlemsini göster
)
# modelin değerlendirilmesi
pred_probs = model.predict(test_gen, verbose = 1)
pred_labels = (pred_probs > 0.5).astype(int).ravel() # eşik değeri 0.5 ile sınıflandırma 
true_labels = test_gen.classes # gerçek etiketler 

cm = confusion_matrix(true_labels, pred_labels) # karışıklık matrisi 
disp = ConfusionMatrixDisplay(cm, display_labels = class_names) 


plt.figure(figsize = (8,8))
disp.plot(cmap = "Blues", colorbar=False)
plt.title("test testi Confusion Matrix") 
plt.show() 
