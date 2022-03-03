
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Dense, Masking, Reshape, Activation, Masking
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, concatenate

def generate_mlstmfcn(max_timesteps, max_features, classes, cells=8):
    # MLSTM-FCN
    # Multivariate Long-Short-Term Memory  
    ip = Input(shape=(max_timesteps, max_features))

    x = Masking()(ip)
    x = LSTM(cells)(x)
    x = Dropout(0.3)(x)

    # y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(classes, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model