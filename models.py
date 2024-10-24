import tensorflow as tf


def build_rnn(hp, vocab_len):
    """
    Build function for RNN with hyperparameter tuning using KerasTuner
    :param hp: HyperModel
    :param vocab_len: vocabulary size
    :return: keras model
    """
    rnn = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=32),
        tf.keras.layers.SimpleRNN(units=hp.Choice('num_units', [32, 64, 128])),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    rnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return rnn


def build_gru(hp, vocab_len):
    """
    Build function for GRU with hyperparameter tuning using KerasTuner
    :param hp: HyperModel
    :param vocab_len: vocabulary size
    :return: keras model
    """
    gru = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=32),
        tf.keras.layers.GRU(units=hp.Choice('num_units', [32, 64, 128, 256])),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    gru.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return gru


def build_bigru(hp, vocab_len):
    """
    Build function for Bidirectional GRU with hyperparameter tuning using KerasTuner
    :param hp: HyperModel
    :param vocab_len: vocabulary size
    :return: keras model
    """
    bigru = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=32),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=hp.Choice('num_units', [32, 64, 128, 256]))),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    bigru.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return bigru


def build_lstm(hp, vocab_len):
    """
    Build function for LSTM with hyperparameter tuning using KerasTuner
    :param hp: HyperModel
    :param vocab_len: vocabulary size
    :return: keras model
    """
    lstm = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=32),
        tf.keras.layers.LSTM(units=hp.Choice('num_units', [32, 64, 128, 256])),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return lstm


def build_bilstm(hp, vocab_len):
    """
    Build function for Bidirectional LSTM with hyperparameter tuning using KerasTuner
    :param hp: HyperModel
    :param vocab_len: vocabulary size
    :return: keras model
    """
    bilstm = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=32),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp.Choice('num_units', [32, 64, 128, 256]))),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    bilstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return bilstm