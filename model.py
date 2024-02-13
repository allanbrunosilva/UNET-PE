
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, ReLU, ZeroPadding2D
from tensorflow.keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = ReLU()(x) # First deconvolution layer followed by ReLU
    
    # Check if padding needs to be added
    if x.shape[1] != skip_features.shape[1] or x.shape[2] != skip_features.shape[2]:
        height_padding = skip_features.shape[1] - x.shape[1]
        width_padding = skip_features.shape[2] - x.shape[2]
        x = ZeroPadding2D(((0, height_padding), (0, width_padding)))(x)
    x = Concatenate()([x, skip_features])

    # Second deconvolution layer followed by ReLU
    x = Conv2DTranspose(num_filters, (3, 3), padding="same")(x)
    x = ReLU()(x)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    s5, p5 = encoder_block(p4, 512)

    b1 = conv_block(p5, 1024)

    d1 = decoder_block(b1, s5, 512)
    d2 = decoder_block(d1, s4, 256)
    d3 = decoder_block(d2, s3, 128)
    d4 = decoder_block(d3, s2, 64)
    d5 = decoder_block(d4, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d5)

    model = Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    input_shape = (364, 364, 5)
    model = build_unet(input_shape)
    model.summary()
