import json
import math
import os
import pickle
import time

import keras
import nltk
import numpy as np
import tensorflow as tf
from nltk.corpus import gutenberg
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Embedding, Dense, LSTM
from keras.optimizers import Adam  # Import Adam optimizer
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten
from keras.layers import MultiHeadAttention
from keras.layers import Input
from keras.models import Model
from keras.layers import GlobalMaxPooling1D


def SLEEP():  # just sleeps 'forever'...  just gives programmer time to stop the run
    time.sleep(9999)


def train_text_generator(N, author):
    data_folder = 'data'
    # Get a list of all subfolders in the data folder
    subfolders = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if
                  os.path.isdir(os.path.join(data_folder, f))]
    for subfolder in subfolders:
        # Get a list of all file paths in the current subfolder
        file_paths = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.txt')]
        # Sort by file size
        file_paths.sort(key=os.path.getsize, reverse=True)
        if subfolder == os.path.join(data_folder, author):  # Handling author's folder
            # Yield the contents of the files that are not in validation or testing
            for i, file_path in enumerate(file_paths):
                loopi = (i % 10) + 1
                # Ignore files that should be used for validation or testing
                if loopi in [6, 7, 8, 9, 10]:
                    continue
                # Yield the contents of the training files, duplicating the files by author N times
                for k in range(N):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        print(file_path)
                        yield file.read().lower()
        else:  # yielding other authors' folders, not duplicating
            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8') as file:
                    #print(file_path)
                    yield file.read().lower()


def valid_text_generator(author):
    author_folder = os.path.join('data', author)
    # Get a list of all file paths in the author's folder
    file_paths = [os.path.join(author_folder, f) for f in os.listdir(author_folder) if f.endswith('.txt')]
    # Sort the file paths by file size, in descending order
    file_paths.sort(key=os.path.getsize, reverse=True)
    # Yield the contents of the files that are not in validation or testing
    for i, file_path in enumerate(file_paths):
        loopi = (i % 10) + 1
        if loopi in [1, 2, 3, 4, 5, 8]:
            continue
        # Yield the contents of the training files
        with open(file_path, 'r', encoding='utf-8') as file:
            print(file_path)
            yield file.read().lower()


def test_text_generator(author):
    author_folder = os.path.join('data', author)
    # Get a list of all file paths in the author's folder
    file_paths = [os.path.join(author_folder, f) for f in os.listdir(author_folder) if f.endswith('.txt')]
    # Sort the file paths by file size, in descending order
    file_paths.sort(key=os.path.getsize, reverse=True)
    # Yield the contents of the files that are not in validation or testing
    for i, file_path in enumerate(file_paths):
        loopi = (i % 10) + 1
        # Ignore files that should be used for validation or testing
        if loopi in [1, 2, 3, 4, 5, 6, 7]:
            continue
        # Yield the contents of the training files
        with open(file_path, 'r', encoding='utf-8') as file:
            print(file_path)
            yield file.read().lower()


def perplexity(model, data_generator):
    total_cross_entropy = 0
    total_samples = 0
    for X_batch, y_batch in data_generator:
        y_pred = model.predict(X_batch)
        batch_cross_entropy = keras.losses.sparse_categorical_crossentropy(y_batch, y_pred)
        total_cross_entropy += np.sum(batch_cross_entropy)
        total_samples += len(y_batch)
    average_cross_entropy = total_cross_entropy / total_samples
    perplexity = math.exp(average_cross_entropy)
    return perplexity


# N is the number of times main text should be duplicated. Large N leads to overfit.
# turns any text from a generator to a np format in batches
def data_generator(tokenizer, batch_size, text_generator):
    # Now we generate sequences in batches
    # text_generator =  train_text_generator(N):
    for text in text_generator:
        sequences_generator = generate_sequences(text, tokenizer)
        X_batch = []
        y_batch = []
        for X, y in sequences_generator:
            X_batch.append(X)
            y_batch.append(y)
            if len(X_batch) >= batch_size:
                yield np.array(X_batch), np.array(y_batch)
                X_batch = []
                y_batch = []
            # if len(X_batch) > batch_size or len(X_batch) < batch_size:
            #    print(f"edge case in data_generator len(X_batch) = {len(X_batch)}   batch_size = {batch_size}")
        # Don't forget to yield what's left if the last batch is smaller than batch_size
        if X_batch:
            yield np.array(X_batch), np.array(y_batch)


def generate_sequences(text, tokenizer):
    tokens = nltk.word_tokenize(text)
    sequences = tokenizer.texts_to_sequences([tokens])[0]
    for i in range(1, len(sequences)):
        yield np.array([sequences[i - 1]]), np.array([sequences[i]])


# uses model to predict text probabilisticly. might be chaotic.
def generate_text_probabilities(model, tokenizer, seed_text, num_words):
    for _ in range(num_words):
        # Convert the input text to sequences of tokens
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = np.array(sequence)
        probabilities = model.predict(sequence)[0]
        predicted_token = np.random.choice(range(len(probabilities)), p=probabilities)
        for word, index in tokenizer.word_index.items():
            if index == predicted_token + 1:
                seed_text += " " + word
                break
    return seed_text


# uses model to predict most likely text by choosing argmax. rigid.
def generate_text_argmax(model, tokenizer, seed_text, num_words):
    for _ in range(num_words):
        # Convert the input text to sequences of tokens
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = np.array(sequence)
        probabilities = model.predict(sequence)[0]
        predicted_token = np.argmax(probabilities)
        for word, index in tokenizer.word_index.items():
            if index == predicted_token + 1:
                seed_text += " " + word
                break
    return seed_text


def generate_predictions(model, authorName, num_words):
    # Load tokenizer
    tokenizer_filename = authorName + "_tokenizer"
    with open(tokenizer_filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    # Use the test text generator to get seed sentences
    test_text_gen = test_text_generator(authorName)
    # Iterate to the 100th seed text
    for _ in range(1):
        next(test_text_gen)
    seed_text = next(test_text_gen)
    # Use only the first sentence as seed
    first_sentence = nltk.sent_tokenize(seed_text)[0]
    full_prediction = generate_text_argmax(model, tokenizer, first_sentence, num_words)
    # full_prediction = generate_text_probabilities(model, tokenizer, first_sentence, num_words)
    # Remove the seed from the prediction
    prediction_only = full_prediction[len(first_sentence):].strip()

    # Get the correct prediction
    tokens = nltk.word_tokenize(seed_text)
    # Get the position of the end of the first sentence
    end_of_seed = tokens.index(nltk.word_tokenize(first_sentence)[-1])
    correct_prediction_tokens = tokens[end_of_seed + 1: end_of_seed + num_words + 1]
    correct_prediction = " ".join(correct_prediction_tokens)

    # Print the seed, its corresponding prediction, and the correct prediction
    print("Seed:", first_sentence)
    print("Prediction:", prediction_only)
    print("Correct Prediction:", correct_prediction)
    print("-----")

    return [full_prediction]


def generate_text_from_test(authorName, folder_path, num_words=20):
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    for model_file in model_files:
        model_path = os.path.join(folder_path, model_file)
        # Load the model
        model = load_model(model_path)
        # Generate predictions
        print(f"Predictions for model: {model_file}")
        generate_predictions(model, authorName, num_words)
        print("=" * 50)  # Add a separator for clarity between different model outputs


def calcAllPerp(authorName, folder_name):
    test_text_gen = test_text_generator(authorName)
    model_files = [f for f in os.listdir(folder_name) if f.endswith('.h5')]
    # Create an empty dictionary to store the results
    perplexity_results = {}
    for model_file in model_files:
        print("Calculating perplexity for model:", model_file)
        tokenizer_filename = authorName + "_tokenizer"
        with open(tokenizer_filename, 'rb') as handle:
            tokenizer = pickle.load(handle)
        test_data_generator = data_generator(tokenizer, 2**11, test_text_gen)  # assuming a batch size of 32
        model = load_model(os.path.join(folder_name, model_file))
        pplx = perplexity(model, test_data_generator)
        print("Perplexity for model", model_file, ":", pplx)
        # Update the dictionary with the results
        perplexity_results[model_file] = pplx
    # Save the dictionary to a file within the folder
    results_filename = os.path.join(folder_name, "perplexity_results.pkl")
    with open(results_filename, 'wb') as handle:
        pickle.dump(perplexity_results, handle)


def avg_sentence_length(text_generator):
    total_length = 0
    total_sentences = 0
    for text in text_generator:
        sentences = nltk.sent_tokenize(text)
        total_sentences += len(sentences)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            total_length += len(words)
    return round(total_length / total_sentences) if total_sentences != 0 else 0


def train_author(author):
    name = author
    tokenizer_filename = author + "_tokenizer"
    # Check if a tokenizer for this author already exists
    if os.path.exists(tokenizer_filename):
        with open(tokenizer_filename, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        training_text = train_text_generator(1, author)  # duplications set to 1 initially
        tokenizer = Tokenizer(oov_token="<OOV>")  # specify OOV token here
        tokenizer.fit_on_texts(training_text)
        with open(tokenizer_filename, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dataset_filename = author + "_dataset_sizes"
    if os.path.exists(dataset_filename):
        with open(dataset_filename, 'r') as file:
            dataset_sizes = json.load(file)
            average_sentence_length = dataset_sizes['average_sentence_length']
            vocab_size = dataset_sizes['vocab_size']
            dup_samples_count = dataset_sizes['dup_samples_count']
            train_total_samples_0dup = dataset_sizes['train_total_samples_0dup']
    else:
        average_sentence_length = avg_sentence_length(train_text_generator(1, author))
        vocab_size = len(tokenizer.word_index) + 1
        train_total_samples1 = 0
        text_generator = train_text_generator(2, author)
        for X_batch, _ in data_generator(tokenizer, 2 ** 5, text_generator):  # just to calculate total samples:
            train_total_samples1 += 2 ** 5 # calc with one duplication
        train_total_samples2 = 0
        text_generator = train_text_generator(3, author)
        for X_batch, _ in data_generator(tokenizer, 2 ** 5, text_generator):  # just to calculate total samples:
            train_total_samples2 += 2 ** 5 # calc with two duplications
        dup_samples_count = train_total_samples2 - train_total_samples1
        train_total_samples_0dup = train_total_samples1 - (2*dup_samples_count)
        print("train_total_samples1:", train_total_samples1)
        print("train_total_samples2:", train_total_samples2)
        print("dup_samples_count:", dup_samples_count)
        print("train_total_samples_0dup:", train_total_samples_0dup)
        with open(dataset_filename, 'w') as file:
            json.dump({
                'average_sentence_length': average_sentence_length,
                'vocab_size': vocab_size,
                'dup_samples_count': dup_samples_count,
                'train_total_samples_0dup':  train_total_samples_0dup,
            }, file)

    print("Average sentence length:", average_sentence_length)
    print("vocab_size = " + str(vocab_size))
    print("dup_samples_count:", dup_samples_count)
    print("train_total_samples_0dup = " + str(train_total_samples_0dup))
    epochs = 10
    for duplications in range(1, 10, 1):
        train_total_samples = train_total_samples_0dup + (duplications * train_total_samples_0dup)
        print("train_total_samples = " + str(train_total_samples))
        train_total_samples = 0
        text_generator = train_text_generator(duplications, author)
        for X_batch, _ in data_generator(tokenizer, 2 ** 5, text_generator):  # just to calculate total samples:
            train_total_samples += 2 ** 5  # calc with two duplications
        print("train_total_samples = " + str(train_total_samples))
        validation_total_samples = 0
        valid_text_gen = valid_text_generator(author)  # changed variable name here
        for X_batch, _ in data_generator(tokenizer, 2 ** 5, valid_text_gen):  # just to calculate total samples:
            validation_total_samples += 2 ** 5
        print("validation_total_samples = " + str(validation_total_samples))

        for test_batch_size in range(5, 7, 1):
            trainRNN(tokenizer, train_total_samples, validation_total_samples, vocab_size, name,
                     epochs, test_batch_size, duplications, average_sentence_length)


def trainRNN(tokenizer, total_samples_train, total_samples_valid, vocab_size, name, epochs,
             test_batch_size, duplications, average_sentence_length):

    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=1))
    model.add(LSTM(50, activation='tanh', input_shape=(None, 1)))
    model.add(Dense(vocab_size, activation='softmax'))
    # Compile your model with Adam optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
    steps_per_epoch = total_samples_train // (2 ** test_batch_size)
    validation_steps = total_samples_valid // (2 ** test_batch_size)

    for epoch in range(epochs):
        # Re-initialize the generators after they were consumed
        text_generator = train_text_generator(duplications, author)
        valid_text_gen = valid_text_generator(author)
        train_generator = data_generator(tokenizer, 2 ** test_batch_size, text_generator)
        valid_generator = data_generator(tokenizer, 2 ** test_batch_size, valid_text_gen)

        # Create a callback that saves the model's weights
        checkpoint = ModelCheckpoint(
            '{}_epoch{:02d}_batch{}_dups{}.h5'.format(name, epoch + 1, test_batch_size, duplications),
            # Python indices start at 0, so we need to add 1 to the epoch
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            mode='auto')

        model.fit(
            x=train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_generator,
            validation_steps=validation_steps,
            epochs=1,
            verbose=1,
            callbacks=[checkpoint]
        )


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    author = 'austen'
    model_path = "MyModels"
    print("train")
    text_generator = train_text_generator(5, author)
    for _ in text_generator:
        pass
    print("valid_________________________")
    text_generator = valid_text_generator(author)
    for _ in text_generator:
        pass
    print("test_________________________")
    text_generator = test_text_generator(author)
    for _ in text_generator:
        pass
    #generate_text_from_test(author, model_path, 20)
    #train_author('austen')  # Train on Austen
    #calcAllPerp(author, folder)
    print("test test test")
    print("test test test")

