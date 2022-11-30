from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from beyondml import tflow
import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings('ignore')

tf.keras.utils.set_random_seed(3892)

(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = tf.keras.datasets.cifar10.load_data()
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = tf.keras.datasets.cifar100.load_data()

cifar10_x_train = cifar10_x_train/255
cifar10_x_test = cifar10_x_test/255
cifar100_x_train = cifar100_x_train/255
cifar100_x_test = cifar100_x_test/255

train_x = np.concatenate((cifar10_x_train, cifar100_x_train), axis = 0)
train_y = np.concatenate((cifar10_y_train, cifar100_y_train), axis = 0).reshape(-1, 1)
test_x = np.concatenate((cifar10_x_test, cifar100_x_test), axis = 0)
test_y = np.concatenate((cifar10_y_test, cifar100_y_test), axis = 0).reshape(-1, 1)
cifar100_train_task_labels = np.asarray([0] * cifar10_x_train.shape[0] + [1] * cifar100_x_train.shape[0]).reshape(-1, 1)
cifar100_test_task_labels = np.asarray([0] * cifar10_x_test.shape[0] + [1] * cifar100_x_test.shape[0]).reshape(-1, 1)
cifar10_train_task_labels = np.asarray([1] * cifar10_x_train.shape[0] + [0] * cifar100_x_train.shape[0]).reshape(-1, 1)
cifar10_test_task_labels = np.asarray([1] * cifar10_x_test.shape[0] + [0] * cifar100_x_test.shape[0]).reshape(-1, 1)

indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)

train_x = train_x[indices]
train_y = train_y[indices]
cifar10_train_task_labels = cifar10_train_task_labels[indices]
cifar100_train_task_labels = cifar100_train_task_labels[indices]

input_layer = tf.keras.layers.Input(train_x.shape[1:])
x = tf.keras.applications.ResNet50(include_top = False)(input_layer)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tflow.layers.MultiMaskedDense(128, activation = 'softmax')([x, x])
x1 = tflow.layers.SelectorLayer(0)(x)
x2 = tflow.layers.SelectorLayer(1)(x)
x1 = tf.keras.layers.Dropout(0.5)(x1)
x2 = tf.keras.layers.Dropout(0.5)(x2)
x1 = tf.keras.layers.BatchNormalization()(x1)
x2 = tf.keras.layers.BatchNormalization()(x2)
x = tflow.layers.MultiMaskedDense(64, activation = 'relu')([x1, x2])
x1 = tflow.layers.SelectorLayer(0)(x)
x2 = tflow.layers.SelectorLayer(1)(x)
x1 = tf.keras.layers.Dropout(0.5)(x1)
x2 = tf.keras.layers.Dropout(0.5)(x2)
x1 = tf.keras.layers.BatchNormalization()(x1)
x2 = tf.keras.layers.BatchNormalization()(x2)
logits = tflow.layers.MultiMaskedDense(100, activation = 'softmax')([x1, x2])

discerner = tflow.layers.MultiMaskedDense(100, activation = 'relu')(logits)
discerner = tflow.layers.MultiMaskedDense(1, activation = 'sigmoid')(discerner)
discerner = tflow.layers.MultitaskNormalization()(discerner)

dsel1 = tflow.layers.SelectorLayer(0)(discerner)
dsel2 = tflow.layers.SelectorLayer(1)(discerner)

lsel1 = tflow.layers.SelectorLayer(0)(logits)
lsel2 = tflow.layers.SelectorLayer(1)(logits)

l1 = tf.keras.layers.Multiply()([dsel1, lsel1])
l2 = tf.keras.layers.Multiply()([dsel2, lsel2])

overall_output = tf.keras.layers.Add()([l1, l2])

test_model = tf.keras.models.Model(input_layer, [overall_output, dsel1, dsel2])
test_model.compile(loss = ['sparse_categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics = ['accuracy'], optimizer = 'adamax')

tf.keras.utils.plot_model(test_model, to_file = 'cifar10_cifar100_model.png', show_shapes = False)

test_model = tflow.utils.mask_model(
    test_model,
    50,
    x = train_x[:1000],
    y = [train_y[:1000], cifar10_train_task_labels[:1000], cifar100_train_task_labels[:1000]]
)
test_model.compile(loss = ['sparse_categorical_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], metrics = ['accuracy'], optimizer = 'adamax')

test_model.fit(
    train_x,
    [train_y, cifar10_train_task_labels, cifar100_train_task_labels],
    epochs = 10,
    batch_size = 256,
    verbose = 1,
    validation_split = 0.2,
    callbacks = tf.keras.callbacks.EarlyStopping(min_delta = 0.004, patience = 3, restore_best_weights = True),
    verbose = 2
)

preds = test_model.predict(test_x)
class_preds = preds[0].argmax(axis = 1).flatten()
cifar10_task_preds = (preds[1] >= 0.5).astype(int).flatten()
cifar100_task_preds = (preds[2] >= 0.5).astype(int).flatten()
test_y = test_y.flatten()

print('Performance for Test Model:')
print('\n')

print('Performance on Identifying CIFAR10 Task:')
print(confusion_matrix(cifar10_test_task_labels, cifar10_task_preds))
print(classification_report(cifar10_test_task_labels, cifar10_task_preds))
print('\n\n')

print('Performance on Identifying CIFAR100 Task:')
print(confusion_matrix(cifar100_test_task_labels, cifar100_task_preds))
print(classification_report(cifar100_test_task_labels, cifar100_task_preds))
print('\n\n')

print('Performance Regardless of Task:')
print(confusion_matrix(test_y, class_preds))
print(classification_report(test_y, class_preds))
print('\n\n')

print('Performance When Truly CIFAR10 Task:')
print(confusion_matrix(test_y[cifar10_task_preds == 1], class_preds[cifar10_task_preds == 1]))
print(classification_report(test_y[cifar10_task_preds == 1], class_preds[cifar10_task_preds == 1]))
print('\n\n')

print('Performance When Truly CIFAR100 Task:')
print(confusion_matrix(test_y[cifar100_task_preds == 1], class_preds[cifar100_task_preds == 1]))
print(classification_report(test_y[cifar100_task_preds == 1], class_preds[cifar100_task_preds == 1]))
print('\n\n')

print('Performance When Predicted CIFAR10 Task:')
print(confusion_matrix(test_y[cifar10_task_preds == 1], class_preds[cifar10_task_preds == 1]))
print(classification_report(test_y[cifar10_task_preds == 1], class_preds[cifar10_task_preds == 1]))
print('\n\n')

print('Performance When Predicted CIFAR100 Task:')
print(confusion_matrix(test_y[cifar100_task_preds == 1], class_preds[cifar100_task_preds == 1]))
print(classification_report(test_y[cifar100_task_preds == 1], class_preds[cifar100_task_preds == 1]))
print('\n\n')

print('Performance When Predicted CIFAR10 Task but Truly CIFAR100 Task:')
try:
    indicator = cifar10_task_preds != cifar10_test_task_labels
    indicator[cifar100_test_task_labels == 0] = 0
    print(confusion_matrix(test_y[indicator], class_preds[indicator]))
    print(classification_report(test_y[indicator], class_preds[indicator]))
except Exception as e:
    print('Not applicable')
print('\n\n')

print('Performance When Predicted CIFAR100 Task but Truly CIFAR10 Task:')
try:
    indicator = cifar100_task_preds != cifar100_test_task_labels
    indicator[cifar10_test_task_labels == 0] = 0
    print(confusion_matrix(test_y[indicator], class_preds[indicator]))
    print(classification_report(test_y[indicator], class_preds[indicator]))
except Exception as e:
    print('Not applicable')
print('\n\n')

print('Overall Performance When Task Incorrectly Predicted:')
try:
    indicator = cifar10_task_preds != cifar10_test_task_labels
    print(confusion_matrix(test_y[indicator], class_preds[indicator]))
    print(classification_report(test_y[indicator], class_preds[indicator]))
except Exception as e:
    print('Not applicable')
print('\n\n')
