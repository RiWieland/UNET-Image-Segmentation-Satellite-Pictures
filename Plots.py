import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss(fitted_model):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(fitted_model.history["loss"], label="loss")
    plt.plot(fitted_model.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(fitted_model.history["val_loss"]), np.min(fitted_model.history["val_loss"]), marker="x",
             color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()


def read_data_jpg(image_dir, batch):
    image_list = random.sample(os.listdir(image_dir), batch)

    # image_list = ['000000229397.jpg', '000000273570.jpg']
    img_np_arr = np.zeros([batch, 128, 128, 3], dtype=np.float32)

    for counter, img_file in enumerate(image_list):
        img = cv2.imread(image_dir + img_file)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

        img_np_arr[counter] = img / 255.

    return img_np_arr


def create_plots_test(model, dict_dir, target_name, batch=7):
    X_data = read_data_jpg(dict_dir['Test'], batch)

    preds = model.predict(X_data)
    preds_t = (preds > 0.5).astype(np.uint8)

    fig, axarr = plt.subplots(3, batch, figsize=(10, 5))

    for i in range(0, batch):
        axarr[0][i + 0].imshow(X_data[i])
        axarr[1][i + 0].imshow(np.squeeze(preds_t[i]), cmap='viridis')
        axarr[2][i + 0].imshow(X_data[i])
        axarr[2][i + 0].imshow(np.squeeze(preds_t[i]), cmap='viridis', alpha=0.1)

    plt.savefig(target_name)
    plt.show()



