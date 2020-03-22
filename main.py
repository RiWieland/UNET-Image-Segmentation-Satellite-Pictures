

import os
from datetime import datetime
import argparse


from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam, Adamax

from Utils import get_dir_dict
from Dataprocessing import create_dataset, generator, create_mask
from Plots import create_plots_test
from Model import model


if __name__ == '__main__':

    dict_dir = get_dir_dict()
    now = datetime.now()

    # Two Modes: 'Create Masks for Training' or 'Train the Network and make Prediction'

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", dest='Mode', type=str ,help="Create Masks for both Training and Validation Set - Mask **or** Train Model and make Predictions -Run",
                        choices=['Mask', 'Run'], required=True)

    parser.add_argument("--Num", type=int, default=1000, help="Number of Masks created for both Training and Validation Set")

    args = parser.parse_args()

    if args.Mode == 'Mask':

        Num_Mask = args.Num

        create_mask('Train', Num_Mask, dict_dir)
        create_mask('Val', Num_Mask, dict_dir)

    if args.Mode == 'Run':

        # create validation set:
        X_val, y_val = create_dataset('Val', 2000, dict_dir)

        # compile
        model = model()
        print(model.summary())

        model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

        Model_Checkpoints = dict_dir['Saved_Models'] + 'Checkpoint_' + now.strftime("%d_%m_%H%M") + '.h5'

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1),
            ModelCheckpoint(Model_Checkpoints, verbose=1, save_best_only=True, save_weights_only=True)
        ]


        # Initialize model
        model.load_weights('model_save_9.h5')

        batch = 8
        results = model.fit_generator(generator('Train', batch, dict_dir), validation_data=(X_val, y_val), steps_per_epoch=130, epochs=1, callbacks=callbacks)

        # Save Model
        model_name = dict_dir['Saved_Models'] + 'model_' + now.strftime("%d_%m_%H%M") + '.h5'
        model.save(model_name)

        # Plot Results
        file_name = os.getcwd() + "/Data/Prediction/Test_" + now.strftime("%d_%m_%H%M") + ".jpg"
        create_plots_test(model, dict_dir, file_name)
