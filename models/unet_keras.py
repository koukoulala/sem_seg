
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

class unet_keras():

    def __init__(self, start_neurons=16,img_size_target=128):
        self.img_size_target = img_size_target
        self.start_neurons = start_neurons
        self.input_layer=Input((img_size_target,img_size_target,1))
        self.output_layer=self.build_model(self.input_layer,16)


    def build_model(self,input_layer, start_neurons):
        # 128 -> 64
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)

        # 64 -> 32
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)

        # 32 -> 16
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)

        # 16 -> 8
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)

        # Middle
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

        # 8 -> 16
        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

        # 16 -> 32
        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

        # 32 -> 64
        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

        # 64 -> 128
        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

        # uconv1 = Dropout(0.5)(uconv1)
        output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

        return output_layer
