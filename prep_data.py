import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def create_vocab(df, column="review", vocab_size=5000):
    """
    Create a vocabulary mapping from words to indices.
    :param df: dataframe containing a column with reviews
    :param column: name of the column containing reviews
    :param vocab_size: maximum size of the vocabulary
    :return: tokenizer object with fitted vocabulary
    """
    # tokenize words
    all_words = []
    for review in df[column]:
        tokens = word_tokenize(review)
        all_words.extend(tokens)
    # create and fit the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<UNK>')
    tokenizer.fit_on_texts(df[column])
    return tokenizer, vocab_size


def prep_review(X, tokenizer, column="review", max_length=None):
    """
    Prepare reviews for later use. Starting off with tokenizing, continuing with encoding using the mapping and padding,
    finishing off by modifying the format to suit TensorFlow/Keras
    :param X: dataframe containing reviews
    :param tokenizer: Tokenizer object with fitted vocabulary
    :param column: name of the column containing reviews
    :param max_length: maximum length for padding sequences (if None, use the length of the longest sequence)
    :return: tf.data.Dataset and indices from the original dataframe to save the predictions
    """
    # tokenize and encode
    sequences = tokenizer.texts_to_sequences(X[column])
    # determine max_length if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    # pad sequences
    padded_reviews = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_reviews, X.index
