import preprocessing
import ml
import pandas as pd
import numpy as np

# Download the data
preprocessing.download_kaggle_data()

# Convert the data to machine learning friendly format
preprocessing.convert_kaggle_csvs_to_npy_files()

# Load data
train_id, train_x, train_y = preprocessing.get_train_data()
valid_id, valid_x, valid_y = preprocessing.get_valid_data()

# Find the most optimal parameters for the model
ml.find_autokeras_model(train_x, train_y, valid_x, valid_y, "spaceship-titanic-model")

# Combine train_x and valid_x as np.array
train_x_valid_x = np.concatenate((train_x, valid_x), axis=0)
# Combine train_y and valid_y as np.array
train_y_valid_y = np.concatenate((train_y, valid_y), axis=0)

# Train the model again with the combined data
ml.train_model(train_x_valid_x, train_y_valid_y, "spaceship-titanic-model", epochs=1000, batch_size=256)

# Load model
model = ml.load_model('spaceship-titanic-model')

# Predict on test data
test_id, test_x = preprocessing.get_test_data()
res = model.predict(test_x)
# Reshape res from 2D to 1D
res = res.reshape(res.shape[0])
# Convert res to boolean where 0 is False and 1 is True
res = [True if x >= 0.5 else False for x in res]

# Prepare submission data with PassengerId from test_id and Transported from res
df = pd.DataFrame(data={'PassengerId': test_id, 'Transported': res})
df.to_csv('spaceship-titanic/submission.csv', index=False)

