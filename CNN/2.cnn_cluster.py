import pickle
from keras.backend.tensorflow_backend import set_session
from sklearn.utils import class_weight

import tensorflow as tf
from sklearn.externals import joblib
from keras import backend as K
from keras.optimizers import Adam, Nadam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout, SpatialDropout2D
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from tensorflow import set_random_seed
from keras.models import load_model
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import argparse
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import pdb
from sklearn.metrics import make_scorer


def create_model(optimizer=SGD,filters = 10, kernel_size=(3, 3), activation='tanh', dropout=0.4,loss='categorical_crossentropy',lr =0.01) :

	"""
	CREATE MODEL WITH MODEL PARAMETERS AS HYPERPARAMTERS TO TUNE
	"""

	K.clear_session()

	model = Sequential([
	    Conv2D(filters = filters, kernel_size = (kernel_size,kernel_size) , activation = activation,
	           data_format = 'channels_last', input_shape = (34, 26, 29)),
	    MaxPooling2D(pool_size = (2,2)),
	    Dropout(rate = dropout),
	    Conv2D(filters = filters, kernel_size = (kernel_size,kernel_size), activation = activation),
	    Dropout(rate = dropout),
	    MaxPooling2D(pool_size = (2,2)),
	    Flatten(),
	    Dense(10, activation = activation),
	    Dense(2, activation ='softmax')
	])


	model.compile(optimizer=optimizer(lr=lr),
				  loss = loss,
				  metrics = [bal_accuracy, sensitivity,specificity ])




	return model

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

POS_CLASS_ID = 1  # Choose the class of interest
NEG_CLASS_ID = 0
def bal_accuracy(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    POS_accuracy_mask = K.cast(K.equal(class_id_preds, POS_CLASS_ID), 'int32')
    POS_class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') *POS_accuracy_mask
    POS_class_acc = K.sum(POS_class_acc_tensor) / K.maximum(K.sum(POS_accuracy_mask), 1)
    
    NEG_accuracy_mask = K.cast(K.equal(class_id_preds, NEG_CLASS_ID), 'int32')
    NEG_class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') *NEG_accuracy_mask
    NEG_class_acc = K.sum(NEG_class_acc_tensor) / K.maximum(K.sum(NEG_accuracy_mask), 1)
    bal_class_acc = 0.5*(POS_class_acc + NEG_class_acc)
    
    return bal_class_acc	



def eval_roc_auc(model): 
	"""
	INPUT : MODEL
	EVALUATE THE MODEL TO GET AUC, ACCURACY, ROC CURVE, FPR, TPR VALUES, ALSO SAVE ROC CURVE 
	"""
	preds = model.predict_proba(test_data)
	#print(preds)
	#print(test_classes)
	auc = roc_auc_score(test_labels, preds[:,1])
        #balacc= balanced_accuracy_score(test_labels, preds[:])) 
	acc = accuracy_score(test_classes[:,1], (preds[:,0]<preds[:,1])*1)
	preds2 = model.predict(test_data)
	balacc2 =balanced_accuracy_score(test_classes[:,1], (preds[:,0]<preds[:,1])*1)
	balacc= balanced_accuracy_score(test_labels, preds2)	
	balacc3 = bal_accuracy(test_labels, preds2)
#	with tf.Session() as sess:
#		init = tf.global_variables_initializer()
#		sess.run(init)
#		balacc3 = bal_accuracy(test_labels, preds2)
	#	print(balacc3)
#		print(balacc3.eval())
#print(preds2)
	#print(test_labels)
	#auc2 = roc_auc_score(test_labels,preds2)
       
 

	# pdb.set_trace()

	print("Test accuracy: ", acc)
	#print("Test accuracy2: ", acc)
	print("Balanced Test accuracy: ",balacc)

	print("Test AUROC: ", auc)
	#print("Test AUROC2: ",auc2)
	print("Balanced Test accuracy2: ",balacc2)

	print("Balanced Test accuracy3: ",balacc3)
	fpr, tpr, thresholds = roc_curve(test_classes[:,1], preds[:,1])

	fig = plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')

	plt.show()
	fig.savefig('/cbica/home/hsuts/figures/ROC Curve.png')


#INSERTING COMMANDLINE OPTIONS FOR THIS SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--TrainDataPath", help="Path to training data pkl",
                    nargs='?', default='/cbica/home/hsuts/data/train_data_laterality.pkl', const=0)
parser.add_argument("--TestDataPath", help="Path to test data pkl",
                    nargs='?', default='/cbica/home/hsuts/data/test_data_laterality.pkl', const=0)
args = parser.parse_args()



#LOADING TRAIN AND TEST DATASETS
with open(args.TrainDataPath, 'rb') as f:
    train_data, train_labels = pickle.load(f)
train_classes = (train_labels)


with open(args.TestDataPath, 'rb') as f:
    test_data, test_labels = pickle.load(f)
test_classes = to_categorical(test_labels)

class_weights = class_weight.compute_class_weight('balanced', np.unique(train_classes),
                                                 train_classes)

#train_data = np.concatenate((train_data, test_data), axis=0)
#train_classes = np.concatenate((train_classes, test_labels), axis=0)

#0 is categorical to [1,0]
#1 is categorical to [0,1]

#HYPERPARAMETER COMBINATIONS
p = {'lr': [0.1, 0.01],
      'filters':[8, 16, 32],
      'batch_size': [2, 16, 32],
      'epochs': [20,40],
      'dropout': [0, 0.4],
      'optimizer': [SGD, Adam, Nadam, RMSprop],
      'loss': ['categorical_crossentropy'],
      'activation':['relu', 'elu', 'tanh'],
      'kernel_size' :[4]
      }


#p = {'lr': [0.01],
#    'filters':[8],
#    'batch_size': [2],
#    'epochs': [20],
#  'dropout': [0.4],
#    'optimizer': [SGD],
#    'loss': ['categorical_crossentropy'],
#   'activation':['elu'],
#   'kernel_size' :[4]

 #   }


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))


seed = 7
np.random.seed(seed)


#if tf.test.gpu_device_name():
#	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#else:
#   	print("Please install GPU version of TF")


#KERAS CLASSIFIER TO USE IN SCIKIT WORKFLOW : Scikit-Learn classifier interface, 
#PREDICT FUNCTION OF SCIKIT AND KERAS WORKS DIFFERENTLY, USE predict_proba() for actual class probabilities

#kerasmodel = KerasClassifier(build_fn=create_model, epochs = 25)
kerasmodel = KerasClassifier(build_fn=create_model)


#tfconfig.gpu_options.allow_growth = True


#grid =  GridSearchCV(estimator = kerasmodel, param_grid = p, cv = 3, scoring= 'roc_auc',refit = True,verbose = 5)


#score = ['accuracy','roc_auc']
score = {'AUC': 'roc_auc', 'Balanced_accuracy': make_scorer(balanced_accuracy_score)}
with tf.device('/device:GPU:1'):
#with tf.device('/cpu:0'):
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#kerasmodel = multi_gpu_model(kerasmodel, gpus=2)
#PERFORMING GRID SEARCH OVER THE HYPERPARAMETER GRID TO SEARCH FOR THE COMBINATION ON BEST ROC_AUC
	grid =  GridSearchCV(estimator = kerasmodel, param_grid = p, cv = 4, scoring= score,refit = 'Balanced_accuracy',verbose = 5)
	#grid =  GridSearchCV(estimator = kerasmodel, param_grid = p, cv = 5, scoring= 'roc_auc',refit = True,verbose = 5)
      #  grid =  GridSearchCV(estimator = kerasmodel, param_grid = p, cv = 3, scoring= 'roc_auc',refit = True,verbose = 5)
	grid_result = (grid.fit(train_data, train_classes,shuffle = 'true', class_weight= class_weights ))
#print(sess.run(grid.fit(train_data, train_classes,shuffle = 'true')))
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	bestmodel = grid.best_estimator_

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#tfconfig = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True )


#sess = set_session( tf.Session(config = tfconfig)) 

#print(sess.run(grid_result))
#b = (sess.run(bestmodel))
#EVALUATE THE BEST MODEL
	eval_roc_auc(bestmodel)

#bestmodel.save('../model/most_recent.h5')
joblib.dump(bestmodel, 'most_recent.joblib')
joblib.dump(bestmodel, 'most_recent.pkl')

# model = load_model('../model/most_recent.h5')



# # Check AUC on the training data, just to verify that the training data was learned.
# score = model.evaluate(train_data, train_classes)
# preds = model.predict(train_data)
# auc = roc_auc_score(train_classes, preds)
# print("Training data accuracy: ", score[1])
# print(f"Training AUROC: {auc}")
# print(len(train_classes))


