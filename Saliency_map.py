from __future__ import print_function
import keras

from keras import activations
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.utils import shuffle


no_classes = 2
validation_split = 0.2
verbosity = 1
batch_size = 32
num_classes = 2
epochs = 100
def mfcc_loader(path='/media/amrgaballah/Backup_Plus/stress/mod_feat11.hdf'):
    data = h5py.File(path)

    X = data['x']
    y = data['y']
    z = data['z']
    print("MSF: %s and feature dimension: %s" % (X.shape[0], X.shape[1]))
    return X, y,z

# the data, split between train and test sets

def run_cnn_mfcc_exp(path ='/media/amrgaballah/Backup_Plus/stress/mod_feat11.hdf'):
    import numpy as np
    # Loading the Digits dataset
    X, y, z = mfcc_loader(path)
    print(X.shape, z.shape)
    X = np.array(X)
    print(X.shape)
    X = X.reshape([X.shape[0], X.shape[1], X.shape[2], 1])
    z = np.array(z)
    z = z.reshape([len(z)])
    X,z = shuffle(X, z, random_state=0)

    #Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.20, random_state=0)


    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 3),
                     activation='relu',
                     input_shape=(23, 8, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='sigmoid', name='visualized_layer'))
    model.summary() 


   

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Fit data to model
    model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    
    # =============================================
    # Saliency Maps code
    # =============================================
    from vis.visualization import visualize_saliency
    from vis.utils import utils
    import matplotlib.pyplot as plt
    import numpy as np
    # Find the index of the to be visualized layer above
    layer_index = utils.find_layer_idx(model, 'visualized_layer')
    # Swap softmax with linear
    model.layers[layer_index].activation = activations.linear
    model = utils.apply_modifications(model) 
    # Numbers to visualize
    indices_to_visualize = [0,10,12,15,25,30,31,35, 38,40,45,47,50,55,59,60,65,80,83,85,90,95,100,105,110,115,120,125,140,112,74,75,190,200,250,257,300,400,500,600,650,700,750,1000,500,800,700,50,100,350,450]
    # Visualize
    map_dic = {0:'low_valence', 1:'high_valence'}
    for index_to_visualize in indices_to_visualize:
        # Get input
        input_image = X_test[index_to_visualize]
        input_class = np.argmax(y_test[index_to_visualize])
        # Matplotlib preparations
        fig, axes = plt.subplots(1, 2)
        # Generate visualization
        visualization = visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)
        axes[0].imshow(input_image[..., 0]) 
        axes[0].set_title('Original image')
        axes[1].imshow(visualization)
        axes[1].set_title('Saliency map')
        out = map_dic.get(input_class)
        print(out)
        fig.suptitle(f'target = {out}')
        plt.show()

    
    
    
    

run_cnn_mfcc_exp()


