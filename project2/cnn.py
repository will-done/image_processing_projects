"""
flowers dataset: 
       rgb: 224*224

cnn ile sınıflandrıma modeli oluşturma ve problemi çözm

"""
# import libraries
from tensorflow_datasets import load 
from tensorflow.data  import AUTOTUNE
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import (
    Conv2D, #2d convolution layer 
    MaxPooling2D, #2d max pooling layer 
    Flatten, #çok boyutlu veriyi tek boyuta indirger 
    Dense, # tam bağlantılı katman
    Dropout # overfitting'i önlemek için 
    )
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import (
    EarlyStopping, #erken durdurma
    ReduceLROnPlateau, # öğrenme oranını azaltma 
    ModelCheckpoint #model kaydetme
 )
import tensorflow as tf 
import matplotlib.pyplot as plt 


# veri seti yükleme
(ds_train, ds_val), ds_info = load(
    "tf_flowers",
    split = ["train[:80%]",
             "train[80%:]"],
    as_supervised = True ,
    with_info = True 
)
print(ds_info.features)
print("number of classes:", ds_info.features["label"].num_classes )

# örnek veri görselleştirme
#eğitim setinden rastgele 3 resim ve etiket alma
fig = plt.figure(figsize = (10,5))
for i, (image, label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1,3,i+1) #1 satır 3 sütun i+1 resim
    ax.imshow(image.numpy().astype("uint8")) #resmi görselleştirme
    ax.set_title(f"etiket: {label.numpy()}") #etiket başlık olarak yazdırma
    ax.axis("off") # eksenleri kapatma
plt.tight_layout()
plt.show() 

IMG_SIZE = (180,180)
# data augmentation + preprocessing
def preprocess_train(image, label):
    """
    resize, random flip, brightness, contrast, crop
    normalize
    """
    image = tf.image.resize (image, IMG_SIZE ) # boyutlandırma
    image = tf.image.random_flip_left_right(image) # rastgele yatay çevirme 
    image = tf.image.random_brightness(image, max_delta=0.1) # parlaklık
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2) # kontrast
    image = tf.image.random_crop(image, size=(160,160, 3)) # rastgele kırpma
    image = tf.image.resize(image, IMG_SIZE ) # tekrar boyutlandırma  
    image = tf.cast(image, tf.float32) / 255.0 # normalize etme 
    return image, label 

def preprocess_val(image, label): 
    """
    resize + normalize
    """
    image = tf.image.resize (image, IMG_SIZE ) # boyutlandırma
    image = tf.cast(image, tf.float32) / 255.0 # normalize etme 
    return image, label

#veri seti hazırlama
ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE) #ön işleme
    .shuffle(1000)
    .batch(32)
    .prefetch(AUTOTUNE) #veri setini önceden hazırlama
)
ds_val = ( 
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE) #ön işleme
    .batch(32)
    .prefetch(AUTOTUNE) #veri setini önceden hazırlama 
)
# cnn modelini oluşturma
model = Sequential([
    #feature extraction layers
    Conv2D(32, (3,3), activation = "relu", input_shape = (*IMG_SIZE,3)), #32 filtre ,3*3 kernel, relu act, 3 kanal rgb
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation = "relu"), # 64 filtre ,3*3 kernel, relu act 
    MaxPooling2D((2,2)), 

    Conv2D(128, (3,3), activation = "relu"), # 128 filtre ,3*3 kernel, relu act 
    MaxPooling2D((2,2)), 

    #classification layers
    Flatten(), #çok boyutlu veriyi tek boyuta indirger 
    Dense(128, activation = "relu"), 
    Dropout(0.5), #overfitting'i önlemek için 
    Dense(ds_info.features["label"].num_classes, activation = "softmax") # çıktı katmanı 



])

# callbacks
callbacks = [
    #eğer val loss 3 epoch boyunca iyileşmezse eğitimi durdur ve en iyi ağırlıkları yükle
    EarlyStopping(
        monitor = "val_loss",
        patience = 3,
        restore_best_weights = True 
    ),
    #val loss 2 epoch boyunca iyileşmezse lr 0.2 ile azalt
    ReduceLROnPlateau(
        monitor = "val_loss",
        factor = 0.2,
        patience = 2,
        verbose = 1,
        min_lr = 1e-9
    ),
    #her epch sonunda eğer model daha iyi ise kaydolur
    ModelCheckpoint(
        "best_model.h5",
        save_best_only = True
    )
]

# derleme
model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss= "sparse_categorical_crossentropy", #etiketler tams ayı olduğu için sparse kullanımı
    metrics = ["accuracy"]

)
print(model.summary()) 

# training
history = model.fit(
    ds_train,
    validation_data = ds_val,
    epochs = 10,
    callbacks = callbacks,
    verbose = 1

)

# model evaluation
plt.figure(figsize = (12,5))
# accuracy plot 
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label = "train accuracy")
plt.plot(history.history["val_accuracy"], label = "val accuracy")
plt.xlabel("Epochs") 
plt.ylabel("Accuracy")
plt.title("Model Accuracy") 
plt.legend()
# loss plot
plt.subplot(1,2,2) 
plt.plot(history.history["loss"], label = "train loss")
plt.plot(history.history["val_loss"], label = "val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss") 
plt.legend()

plt.tight_layout() 
plt.show() 

