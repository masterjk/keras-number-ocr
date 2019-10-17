import tensorflow as tf

# Step 1: Import Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Step 2: Build Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Reshape
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Step 4: Convert class vector (int) to binary class matrix
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Step 5: Train it
model.fit(train_images, train_labels, batch_size=128, epochs=5, verbose=1)

# Step 6: Perform the evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Step 7: Save the model
model.save("mnist.h5")

# Step 8: Debug
model.summary()
print(model.to_json())
