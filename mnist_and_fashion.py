from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from beyondml import tflow
import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings('ignore')

tf.keras.utils.set_random_seed(3892)

(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = tf.keras.datasets.mnist.load_data()
(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()

train_x = np.concatenate((mnist_x_train, fashion_x_train), axis = 0)/255
train_y = np.concatenate((mnist_y_train, fashion_y_train), axis = 0)
test_x = np.concatenate((mnist_x_test, fashion_x_test), axis = 0)/255
test_y = np.concatenate((mnist_y_test, fashion_y_test), axis = 0)
mnist_train_labels = np.asarray([1] * mnist_x_train.shape[0] + [0] * fashion_x_train.shape[0])
fashion_train_labels = np.asarray([0] * mnist_x_train.shape[0] + [1] * fashion_x_train.shape[0])
mnist_test_labels = np.asarray([1] * mnist_x_test.shape[0] + [0] * fashion_x_test.shape[0])
fashion_test_labels = np.asarray([0] * mnist_x_test.shape[0] + [1] * fashion_x_test.shape[0])

indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)

train_x = train_x[indices]
train_y = train_y[indices]
mnist_train_labels = mnist_train_labels[indices]
fashion_train_labels = fashion_train_labels[indices]

input_layer = tf.keras.layers.Input(train_x.shape[1:])
x = tf.keras.layers.Flatten()(input_layer)
x = tflow.layers.MultiMaskedDense(100, activation = 'relu')([x, x])
for _ in range(4):
    x = tflow.layers.MultiMaskedDense(100, activation = 'relu')(x)
logit_output = tflow.layers.MultiMaskedDense(10, activation = 'softmax')(x)

sel1 = tflow.layers.SelectorLayer(0)(logit_output)
sel2 = tflow.layers.SelectorLayer(1)(logit_output)
discerner = tflow.layers.MultiMaskedDense(10, activation = 'relu')([sel1, sel2])
discerner = tflow.layers.MultiMaskedDense(1, activation = 'sigmoid')(discerner)
discerner = tflow.layers.MultitaskNormalization()(discerner)
dsel1 = tflow.layers.SelectorLayer(0)(discerner)
dsel2 = tflow.layers.SelectorLayer(1)(discerner)

discerner_x1 = tf.keras.layers.Multiply()([sel1, dsel1])
discerner_x2 = tf.keras.layers.Multiply()([sel2, dsel2])

output = tf.keras.layers.Add()([discerner_x1, discerner_x2])

test_model = tf.keras.models.Model(input_layer, [output, dsel1, dsel2])
test_model.compile(loss = ['sparse_categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics = ['accuracy'], optimizer = 'adam')

test_model = tflow.utils.mask_model(
    test_model,
    50,
    x = train_x[:100],
    y = [train_y[:100], mnist_train_labels[:100].reshape(-1, 1), fashion_train_labels[:100].reshape(-1, 1)]
)
test_model.compile(loss = ['sparse_categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics = ['accuracy'], optimizer = 'adam')

tf.keras.utils.plot_model(test_model, to_file = 'mnist_fashion_model.png', show_shapes = False)

test_model.fit(
    train_x,
    [train_y, mnist_train_labels, fashion_train_labels],
    epochs = 100,
    batch_size = 128,
    verbose = 0,
    validation_split = 0.2,
    callbacks = tf.keras.callbacks.EarlyStopping(min_delta = 0.004, patience = 3)
)

preds = test_model.predict(test_x)
class_preds = preds[0].argmax(axis = 1).flatten()
pred_mnist = (preds[1]>= 0.5).astype(int).flatten()
pred_fashion = (preds[2]>=0.5).astype(int).flatten()

print('Performance for Test Model:')
print('\n')

print('Performance on Identifying Digit Task:')
print(confusion_matrix(mnist_test_labels, pred_mnist))
print(classification_report(mnist_test_labels, pred_mnist))
print('\n\n')

print('Performance on Identifying Fashion Task:')
print(confusion_matrix(fashion_test_labels, pred_fashion))
print(classification_report(fashion_test_labels, pred_fashion))
print('\n\n')

print('Performance Regardless of Task:')
print(confusion_matrix(test_y, class_preds))
print(classification_report(test_y, class_preds))
print('\n\n')

print('Performance When Truly Digit Task:')
print(confusion_matrix(test_y[mnist_test_labels == 1], class_preds[mnist_test_labels == 1]))
print(classification_report(test_y[mnist_test_labels == 1], class_preds[mnist_test_labels == 1]))
print('\n\n')

print('Performance When Truly Fashion Task:')
print(confusion_matrix(test_y[fashion_test_labels == 1], class_preds[fashion_test_labels == 1]))
print(classification_report(test_y[fashion_test_labels == 1], class_preds[fashion_test_labels == 1]))
print('\n\n')

print('Performance When Predicted Digit Task:')
print(confusion_matrix(test_y[pred_mnist == 1], class_preds[pred_mnist == 1]))
print(classification_report(test_y[pred_mnist == 1], class_preds[pred_mnist == 1]))
print('\n\n')

print('Performance When Predicted Fashion Task:')
print(confusion_matrix(test_y[pred_fashion == 1], class_preds[pred_fashion == 1]))
print(classification_report(test_y[pred_fashion == 1], class_preds[pred_fashion == 1]))
print('\n\n')

print('Performance When Predicted Digit Task but Truly Fashion Task:')
try:
    indicator = pred_mnist != mnist_test_labels
    indicator[fashion_test_labels == 0] = 0
    print(confusion_matrix(test_y[indicator], class_preds[indicator]))
    print(classification_report(test_y[indicator], class_preds[indicator]))
except Exception as e:
    print('Not applicable')
print('\n\n')

print('Performance When Predicted Fashion Task but Truly Digit Task:')
try:
    indicator = pred_fashion != fashion_test_labels
    indicator[mnist_test_labels == 0] = 0
    print(confusion_matrix(test_y[indicator], class_preds[indicator]))
    print(classification_report(test_y[indicator], class_preds[indicator]))
except Exception as e:
    print('Not applicable')
print('\n\n')

print('Overall Performance When Task Incorrectly Predicted:')
try:
    indicator = pred_mnist != mnist_test_labels
    print(confusion_matrix(test_y[indicator], class_preds[indicator]))
    print(classification_report(test_y[indicator], class_preds[indicator]))
except Exception as e:
    print('Not applicable')
print('\n\n')
