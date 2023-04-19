import numpy as np
import matplotlib.pyplot as plt  
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import time

# Hide GPU from visible devices
tf.config.set_visible_devices([], "GPU")

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# model_name = "non_pruned.h5"
model_name = "pruned_model.h5"
image_name = "4.jpg"
image_label = 4


image = Image.open(image_name).convert("L")
image = image.resize((28, 28), Image.Resampling.LANCZOS)
img = np.array(image)
img = 255 - img

im = img.astype("float32") / 255.0


def plot_image(predictions_array, true_label, imgarg):
    plt.title(f"Model:{model_name}")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(imgarg, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% (label:{})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label],
        ),
        color=color,
    )


def plot_value_array(predictions_array, true_label, latency):
    plt.title(f"Latency:{latency}ms")
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


model = load_model(model_name)
model.predict(im[None, :, :])
t1 = time.time()
predict = model.predict(im[None, :, :])
t2 = time.time()
t = round(1000 * (t2 - t1), 2)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(predict, image_label, img)
plt.subplot(1, 2, 2)
plot_value_array(predict[0], image_label, t)
plt.show()