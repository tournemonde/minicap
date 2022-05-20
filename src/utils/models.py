from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG19, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  GlobalAveragePooling2D, BatchNormalization, Conv2D, Flatten, MaxPool2D, Dropout, ReLU, Input
from tensorflow.keras.models import Model


def build_efficientnet(num_classes):
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    
    # Freeze layers
    base_model.trainable = False

    model = Sequential(name="Vehicle_Recognition")
    model.add(base_model)
    model.add(GlobalAveragePooling2D(name="GAP"))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax", name="Probs"))

    return model


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


def build_cnn(num_classes):
    model = Sequential(name='cnn')
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='valid', input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D(name="GAP"))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_vgg19(num_classes, train_blocks=None):
    model = Sequential(name='vgg19-vehicle-counter')
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    if train_blocks is None:
        base_model.trainable = False
    elif isinstance(train_blocks, list):
        for tb in train_blocks:
            for layer in base_model.layers:
                if tb in layer.name:
                    layer.trainable = True
                else:
                    layer.trainable = False
    elif train_blocks=='all':
        base_model.trainable=True
    else:
        for layer in base_model.layers:
            if train_blocks in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

    model.add(base_model)
    model.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D(name="GAP"))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_vgg19_functional(num_classes, train_blocks=None, pure=False):

    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    if train_blocks is None:
        base_model.trainable = False
    elif isinstance(train_blocks, list):
        for layer in base_model.layers:
            exist = [1 if block in layer.name else 0 for block in train_blocks ]
            if any(exist):
                layer.trainable = True
            else:
                layer.trainable = False
    elif train_blocks == 'all':
        base_model.trainable = True
    else:
        for layer in base_model.layers:
            if train_blocks in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

    if not pure:
        head = Conv2D(512, kernel_size=3, activation='relu', padding='same')(base_model.output)
        head = BatchNormalization()(head)
        head = MaxPool2D(pool_size=(2, 2))(head)
        head = Conv2D(256, kernel_size=3, activation='relu', padding='same')(head)
        head = BatchNormalization()(head)
        head = MaxPool2D(pool_size=(2, 2))(head)
        head = GlobalAveragePooling2D(name="GAP")(head)
    else:
        head = GlobalAveragePooling2D(name="GAP")(base_model.output)

    head = Dropout(0.1)(head)
    head = Dense(128)(head)
    head = BatchNormalization()(head)
    head = ReLU()(head)
    head = Dropout(0.1)(head)
    head = Dense(num_classes, activation='softmax')(head)

    model = Model(inputs=base_model.inputs, outputs=head)

    return model


def build_mnetv2(num_classes, image_shape=(224, 224, 3), fine_tune_layer=None):
    base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    print(f"Number of layers in the base model: {len(base_model.layers)}")

    if fine_tune_layer is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_layer]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation='relu')(x)    
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model
