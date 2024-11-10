
import os
import numpy as np
import pandas as pd
import joblib  # Import joblib for saving the encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense

# Set to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Importing the dataset
try:
    dataset = pd.read_csv('updated_test.csv')
except FileNotFoundError:
    print("The specified file was not found.")
    raise

# Data preprocessing
dataset = dataset[dataset['YEAR'] > 1980]
dataset = dataset.dropna()

# Features and Target
X = dataset.iloc[:, [0, 3, 4, 6]].values  # Adjust indices based on your dataset
y = dataset.iloc[:, 5].values  # Adjust index based on your target column

# Encoding categorical data
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])  # Encode city

labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X2.fit_transform(X[:, 1])  # Encode state

labelencoder_X3 = LabelEncoder()
X[:, 3] = labelencoder_X3.fit_transform(X[:, 3])  # Encode terrain

# OneHotEncoder using ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), [0, 1, 3])  # Specify categorical features here
    ],
    remainder='passthrough'  # Keep other columns as they are
)
X = column_transformer.fit_transform(X)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Save the fitted label encoders
joblib.dump(labelencoder_X1, 'labelencoder_X1.pkl')
joblib.dump(labelencoder_X2, 'labelencoder_X2.pkl')
joblib.dump(labelencoder_X3, 'labelencoder_X3.pkl')
joblib.dump(labelencoder_y, 'labelencoder_y.pkl')  # Save target label encoder

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# One-hot encode y_train without sparse argument
y_train = np.reshape(y_train, (-1, 1))
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()

# Feature Scaling
sc_X = StandardScaler(with_mean=False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Using Neural Networks
# Initialising the ANN
classifier = Sequential()

# Adding layers
classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=y_train.shape[1], kernel_initializer='uniform', activation='softmax'))  # Match output units to number of classes

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=500, epochs=500, validation_split=0.2)  # Use 'epochs' set to 10

# Predicting the Test set results
y_pred = classifier.predict(X_test)
res = np.argmax(y_pred, axis=1)  # Get the predicted class

# Evaluate the model
accuracy = accuracy_score(y_test, res)
print(f"Accuracy: {accuracy:.2f}")

# Generate classification report
print(classification_report(y_test, res))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, res)
print("Confusion Matrix:\n", cm)

# Saving the model
model_json = classifier.to_json()
with open("./model/flood_model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("./model/flood_model.weights.h5")
print("Saved model to disk")
