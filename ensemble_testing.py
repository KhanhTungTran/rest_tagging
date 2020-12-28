import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy as np
from sklearn.metrics import accuracy_score


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = np.array(yhats)
	# sum across ensemble members
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result

def evaluate(model_paths, preprocess_batch_path):
    model_paths = []
    models = list(map(lambda a: keras.models.load_model(a), model_paths))
    valids = np.load(preprocess_batch_path + '/preprocess_validation.npz')
    yhat = ensemble_predictions(
        models, valids["features"])
    return accuracy_score(valids["labels"], yhat)

if __name__ == "__main__":
    model_paths = [] # NOTE: add path to models here, this should be a list of string
    print(evaluate(model_paths, 'Preprocess_batch'))
