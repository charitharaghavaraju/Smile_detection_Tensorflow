import numpy as np
from data_generator import data_loader
from sklearn.model_selection import train_test_split
from model import build_model
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main(data_dir, train_flag):

    images, labels = data_loader(data_dir)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.20, train_size=0.8, random_state=42)

    if train_flag:
        model = build_model()
        model.summary()

        model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
        ckpt = ModelCheckpoint(filepath='best_model2.h5', verbose=0, monitor='val_accuracy', mode='max', save_best_only=True)

        history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels), callbacks=[ckpt])

        # plot and save the accuracy graph
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig('accuracy.jpeg')

    else:
        model = build_model()
        model.load_weights('best_model2.h5')  # load pretrained weights
        pred = model.predict(test_images)  # predict on test data

        # calculate test accuracy and confusion matrix
        pred = np.reshape(pred, -1)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        validation_accuracy = 1 - np.count_nonzero(pred - test_labels) / len(pred)
        conf_matrix = ConfusionMatrixDisplay(confusion_matrix(test_labels, pred))
        conf_matrix.plot()
        plt.show()
        print('Accuracy on test data = ', validation_accuracy)


if __name__ == '__main__':
    dataset_dir = "GENKI-4K"
    train = False  # True for training mode
    main(dataset_dir, train)
