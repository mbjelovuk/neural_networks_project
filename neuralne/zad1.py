import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras_tuner as kt

def make_model(hp):
    model = Sequential()
    model.add(Dense(
        units=hp.Choice('neurons', values=[16, 32, 64]),
        input_dim=8,
        activation='relu'
    ))
    model.add(Dropout(hp.Choice('dropout_rate', values=[0.0, 0.2, 0.5])))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.01, 0.1]))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


data = pd.read_csv('./Star.csv')

nun = [col for col in data.columns if col.strip() == '' or 'Unnamed' in col]
data = data.drop(columns=nun)

klase = ['0 - nije zvezda', '1 - jeste']
bru = [4213, 1639]
plt.bar(klase, bru)
plt.title('Broj uzoraka po klasama')
plt.xlabel('Klase')
plt.ylabel('Broj uzoraka')
plt.show()

X = data.drop('Class', axis=1).values
Y = data['Class'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

es = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
tuner = kt.RandomSearch(
    make_model,
    objective='val_accuracy',
    overwrite=True,
    max_trials=5
)
tuner.search(X_train, Y_train,
             epochs=100,
             batch_size=32,
             callbacks=[es],
             validation_split=0.2,
             verbose=3
)
best_hyperparam = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Najbolji hiperparametri:", best_hyperparam.values)

finalnimod = tuner.hypermodel.build(best_hyperparam)
history = finalnimod.fit(X_train, Y_train,
                         epochs=50,
                         batch_size=64,
                         validation_split=0.2,
                         verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

Y_pred = (finalnimod.predict(X_test) > 0.5).astype(int)

acc = accuracy_score(Y_test, Y_pred)
prec = precision_score(Y_test, Y_pred)
rec = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
rocauc = roc_auc_score(Y_test, Y_pred)

print(f"Tačnost: {acc}")
print(f"Preciznost: {prec}")
print(f"Osetljivost (Recall): {rec}")
print(f"F1-skor: {f1}")
print(f"roc auc: {rocauc}")

plt.figure()
cm_train = confusion_matrix(Y_train, np.round(finalnimod.predict(X_train)), normalize='true')
cm_display_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=klase)
cm_display_train.plot()
plt.title('Confusion Matrix - Training Set', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure()
cm_test = confusion_matrix(Y_test, Y_pred, normalize='true')
cm_display_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=klase)
cm_display_test.plot()
plt.title('Confusion Matrix - Test Set', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()
