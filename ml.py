"""
Machine learning module.
"""

import autokeras as ak
import tensorflow as tf
from keras.models import load_model as lm

def find_autokeras_model(train_x, train_y, val_x, val_y, model_name):
    """
    Use Autokeras to find most optimal structured data classifier architecture for the task.
    """
    clf = ak.StructuredDataClassifier(project_name=f'{model_name}-autokeras', seed=42, max_trials=10)
    clf.fit(train_x, train_y, validation_data=[val_x, val_y])
    model = clf.export_model()
    model.save(model_name, overwrite=True, save_format='tf')
    return clf

def train_model(X, y, model_name, epochs=10, batch_size=32):
    """
    Train a model.
    """
    model = load_model(model_name)
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    model.save(model_name, overwrite=True, save_format='tf')
    return model

def load_model(model_name) -> tf.keras.models.Model:
    """
    Load a model.
    return an object of type tf.keras.models.Model
    """
    loaded_model = lm(model_name, custom_objects=ak.CUSTOM_OBJECTS)
    return loaded_model