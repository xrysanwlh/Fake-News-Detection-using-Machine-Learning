import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model
from time import time
import visualkeras
from DataPreProccess import test_text, test, X_train, X_test, y_train, y_test, vocab_size, embeddings_matrix, maxFeatures
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model():
    # Build the architecture of the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size+1, output_dim = 100, weights=[embeddings_matrix],input_length=maxFeatures),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    return model


def train():

    start = time()

    if not os.path.exists("model.h5"):
        model = create_model()
    else:
        print("Loading and evaluating  model")
        model = load_model('model.h5')
        

    model.summary()
    plot_model(model, to_file='model_plot.png', show_layer_names=True)
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    model.save('model.h5')
    print("Saved  model to disk")
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train:', train_acc, 'Test:', test_acc)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    print("time = \n", time()-start)

    return history


def predict():

    model = load_model('model.h5')
    prediction = model.predict(test_text)
    prediction = np.reshape(prediction, -1)
    prediction = np.round(prediction)
    y_testAccuracy = np.asarray(test['Overall_Rating'])
    accuracy = accuracy_score(list(y_testAccuracy), prediction)
    print("Model Accuracy : ", accuracy)

    return prediction


if __name__ == "__main__":
    print("\t1.Train")
    print("\t2.Predict\n")
    choice = input("your choice? ")
    choice = int(choice)

    if choice == 1:
        model_fit = train()
    elif choice == 2:
        prediction = predict()
        cm = confusion_matrix(list(test['Overall_Rating']), prediction)
        sns.heatmap(cm, annot = True)
        plt.title('Heatmap of Fakenews Dataset')
        plt.xlabel('supposed to')
        plt.ylabel('prediction')
        print(classification_report(test['Overall_Rating'], prediction))
        plt.show()
        pd_pred = pd.DataFrame({'id': test.index, 'prediction': prediction})
        pd_pred.to_csv('prediction.csv', index=False)
    else:
        print("Invalid input")
