import matplotlib.pyplot as plt
from os import makedirs
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense


def build_resnet50(num_classes, trainable=False):
    model = Sequential()

    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    # weights = r'/data/04_pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(num_classes, activation='softmax'))

    # Say not to train first layer (ResNet) model as it is already trained
    if not trainable:
        model.layers[0].trainable = False
    
    return model


def plot_hist(hist, name):
    plt.clf()
    plt.plot(hist.history["precision"])
    plt.plot(hist.history["val_precision"])
    plt.title("model precision")
    plt.ylabel("precision")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

    makedirs('data/04_report/', exist_ok=True)
    plt.savefig(f"data/04_report/{name}_history.png")


def plot_fit_hist(hist, model_name):
    plt.clf()
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model fit")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

    makedirs('data/04_report/', exist_ok=True)
    plt.savefig(f"data/04_report/{model_name}_fit_history.png")


# # TODO nicer and annotate
# def plot_cm(cm, classes: list):
#     plt.clf()
#     plt.figure(figsize=(10, 10))
#     ax = plt.gca()
#     ax.matshow(cm)
#     ax.set_xticks(range(len(classes)))
#     ax.set_xticklabels(classes, fontdict={'fontsize': 6})
#     plt.setp(ax.get_xticklabels(), rotation=65, ha="left", rotation_mode="anchor")
#     ax.set_yticks(range(len(classes)))
#     ax.set_yticklabels(classes, fontdict={'fontsize': 6})
#     plt.setp(ax.get_yticklabels(), rotation=25, ha="right", rotation_mode="anchor")

#     plt.savefig(join(REPORT_FOLDER, MODEL_NAME + '_cm.png'))


# def evaluation_metrics(labels, preds, classes, plot=True):
#     cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))
#     print(cm)

#     prec = precision_score(labels, preds, labels=range(NUM_CLASSES), average='macro', zero_division=0)
#     rec = recall_score(labels, preds, labels=range(NUM_CLASSES), average='macro', zero_division=0)
#     f1 = f1_score(labels, preds, labels=range(NUM_CLASSES), average='macro', zero_division=0)

#     print('Precision \t: {:.4f}'.format(prec))
#     print('Recall \t\t: {:.4f}'.format(rec))
#     print('F1 score \t: {:.4f}'.format(f1))

#     if plot:
#         plot_cm(cm, classes)

