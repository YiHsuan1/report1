# report1
南華大學跨領域-人工智慧期中報告 11123024楊佳宜 11123007陳奕瑄

導入相關的包
!pip install tensorflow keras numpy mnist matplotlib
#導入數據包
mport numpy as np
import mnist  # 获得数据集
import matplotlib.pyplot as plt  # Graph
from keras.models import Sequential  # ANN 网络结构
from keras.layers import Dense # the layer in  the  ANN
import keras
import keras.utils
from keras import utils as np_utils

導入mnist數據集中對應數據
# 導入數據
train_images = mnist.train_images()  # 訓練數據集圖片
train_labels = mnist.train_labels()   # 訓練標籤
test_images = mnist.test_images()  # 測試圖片
test_labels = mnist.test_labels()  # 測試標籤

對數據進行相應的處理，將圖片數據歸一化，同時向量化
# 規範化圖片 規範化像素值[0,255]
# 為了讓神經網路更好的訓練，我們把數值設定為[-0.5 , 0.5]
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5
# 將 28 * 28 像素圖片展成 28 * 28 = 784 維向量
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))
#列印出來
print(train_images.shape) # 6000個訓練數據
print(test_images.shape) # 1000個測試數據

建立神經網絡模型
在這裏我們為了方面初學者，我們使用 Keras Sequential 順序模型 順序模型是多個網絡層的線性堆疊。 你可以通過將網絡層實例的列表傳遞給 Sequential 的構造器，來創建一個 Sequential 模型
# 建立模型
# 3層 ，其中兩層 64 個神經元 以及激勵函數  一層10個神經元 以及歸一化指數函數（softmax fuction）
model = Sequential()
model.add( Dense(64, activation="relu", input_dim = 784))
model.add( Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
print(model.summary())

進行模型的編譯和訓練
# 編譯模型 
# 損失函數衡量模型在訓練中的表現 然後進行優化
model.compile(
    optimizer = 'adam',
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)
# 訓練模型
from keras.utils.np_utils import to_categorical
history=model.fit(
    train_images,
    to_categorical(train_labels),
    epochs = 5,  #要訓​​練的整個數據集的疊代次數
    batch_size = 32  #每個梯度更新的樣本數以進行訓練

)

print(history.history.keys())
# print(plt.plot(history.history['loss']))
print(plt.plot(history.history['accuracy']))

評估模型
# 評估模型
model.evaluate(
    test_images,
    to_categorical(test_labels)
)

進行預測
# 保存模型
# 預測前五個圖片


predictions = model.predict(test_images[:5])
# 輸出模型預測 同時和標準值進行比較
print(np.argmax(predictions, axis = 1))
print(test_labels[:5])

看看在mnist中的圖片長怎樣
for i in range(0,5):
  first_image = test_images[i]
  first_image = np.array(first_image ,dtype= "float")
  pixels = first_image.reshape((28 ,28))
  plt.imshow(pixels , cmap="gray")
  plt.show()
  
識別自己的手寫體
首先需要建立連結： 因為colab 的文件在雲端中，我們需要它和google drives 進行綁定

import os
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive/data"

os.chdir(path)
os.listdir(path)
之後進行模型預測

rom PIL import Image
import numpy as np
import os

img = Image.open("test.jpg").convert("1")
img = np.resize(img, (28,28,1))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,784)
y_pred = model.predict(im2arr)
print(np.argmax(y_pred, axis = 1))

參考文章
https://blog.csdn.net/weixin_43843172/article/details/109897787
