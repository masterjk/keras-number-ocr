import tensorflow as tf

# Step 1: Import Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Step 2a: Build Network
# Chapter 2: https://my.safaribooksonline.com/book/programming/python/9781617294433/chapter-2dot-before-we-begin-the-mathematical-building-blocks-of-neural-networks/ch02lev1sec5_html
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
# train_labels = tf.keras.utils.to_categorical(train_labels, 10)
# test_labels = tf.keras.utils.to_categorical(test_labels, 10)
# model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Step 2b: Build Network
# Chapter 5: https://my.safaribooksonline.com/9781617294433/ch02lev1sec5_html
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
model.fit(train_images, train_labels, batch_size=64, epochs=5, verbose=1)

# Step 6: Perform the evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Step 7: Save the model
model.save("mnist.h5")

# Step 8: Debug
model.summary()
print(model.to_json())
