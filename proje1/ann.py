"""
MNIST veri seti:
    rakamlama: 0-9 toplamda 10 sınıf var
    28*28 piksel boyutunda resimler
    grayscale siyah-beyazz resimler
    60000 eğitim, 10000 test verisi
    amaç:ann ile bu resimleri tanımlamak ya da sınıflandırmak

    image processing:
        histogram eşitleme
        gaussian blur 
        canny edge detection 

    ANN (Artificial Neural Network): ile MNIST veri setini sınıflandırmak 

"""
# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt  

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam 

#load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data( )
print(f"x_train shape:{x_train.shape}")
print(f"y_train shape:{y_train.shape}") 

"""
x_train shape:(60000, 28, 28) 
y_train shape:(60000,) 
"""

#image preprocessing
img = x_train[5]  #ilk resmi alalım 
stages = {"original":img}
#histogram equalization 
eq =  cv2.equalizeHist(img) 
stages["histogram equalization"] = eq 
#gaussian blur : gürültüyü azlatm
blur = cv2.GaussianBlur(eq, (5,5),0)
stages["gaussian blur"] = blur
#canny edge detection 
edges = cv2.Canny(blur,50,150)  #kenar tespiti
stages["canny edge detection"] = edges 

#görselleştirme
fig, axes = plt.subplots(2,2, figsize=(6,6)) 
axes =axes.flat
for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off") 

plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.show()

#preprocessing fonksionu
def preprocess_images(img):
    """
    histogram eşitleme
    guassian blur
    canny ile kenar tespiti
    flattering 28*28 boyutundan 784e çevrme
    normalizasyon 0-255 arası 0-1 arası yapma
    """
   
    img_eq = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eq, (5,5),0)
    img_edges = cv2.Canny(img_blur,50,150)
    features = img_edges.flatten()/255.0  #28*28=784 
    return features

num_train = 10000
num_test= 2000

x_train = np.array([preprocess_images(img) for img in x_train[:num_train]]) 
y_train_sub = y_train[:num_train] 

x_test = np.array([preprocess_images(img) for img in x_test[:num_test]]) 
y_test_sub = y_test[:num_test]

       




# #ann model creation
model = Sequential(
    [
       Dense(128, activation="relu", input_shape=(784,)),
       Dropout(0.5),
       Dense(64, activation="relu"),#ikinci katman
       Dense(10, activation="softmax")#çıkış katmanı
    ] 
)

#compile model
model.compile(
    optimizer=Adam(learning_rate=0.001 ),
    loss="sparse_categorical_crossentropy", #kayıp fonk
    metrics=["accuracy"]
) 

#model summary 
print (model.summary())
#ann model training 
history = model.fit( 
    x_train, y_train_sub,
    validation_data=(x_test, y_test_sub), 
    epochs=50,
    batch_size=32,
    verbose = 2
   
) 

# #evaluate and preformance
test_lost , test_acc = model.evaluate(x_test, y_test_sub) 
print(f"Test Loss: {test_lost:.4f}, Test Accuracy: {test_acc:.4f} ") 

#plot training history 
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.title('Loss ') 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend( )

plt.tight_layout( )
plt.show( )

