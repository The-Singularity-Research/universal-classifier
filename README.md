# universal-classifier
Single qubit data reuploading universal binary classifier modified from Penny Lane

These two notebooks are modified versions of the [Data Reuploading Classifier](https://pennylane.ai/qml/app/tutorial_data_reuploading_classifier.html) from [Penny Lane](https://pennylane.ai/). One runs on the sckikit-learn data set 'make_moons', the other runs on the scikit-learn data set 'make_circles'. 

### Scitkit-Learn Classifier Comparison
![alt text](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

### Data Reuploading Classifier Traning Data on make_moons

#### Generating the Data
'''# Generate training and test data
num_training = 200
num_test = 2000

Xdata, y_train = make_moons(num_training, noise=0.1)
X_train = np.hstack((Xdata, np.zeros((Xdata.shape[0], 1))))

Xtest, y_test = make_moons(num_test, noise=0.1)
X_test = np.hstack((Xtest, np.zeros((Xtest.shape[0], 1))))'''

#### Epochs and Loss Function
'''num_layers = 3
learning_rate = 0.6
epochs = 20
batch_size = 32'''

![traning_data_image_make_moons](traning_data_image_make_moons.png)

### Data Reuploading Classifier Traning Data on make_circles

#### Generating the Data
'''# Generate training and test data
num_training = 200
num_test = 2000

Xdata, y_train = make_circles(num_training, noise=0.1)
X_train = np.hstack((Xdata, np.zeros((Xdata.shape[0], 1))))

Xtest, y_test = make_circles(num_test, noise=0.1)
X_test = np.hstack((Xtest, np.zeros((Xtest.shape[0], 1))))'''

#### Epochs and Loss Function
'''num_layers = 3
learning_rate = 0.6
epochs = 20
batch_size = 32'''




![traning_data_image_make_circles](traning_data_image_make_circles.png)





