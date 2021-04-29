import random
import json
import re
import string
import sys
import datetime as dt
import nltk as nl
import matplotlib.pyplot as plt
import graph_data_crawler
import extra_graphs
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import classify, NaiveBayesClassifier


# Replaces similar words in the list of tokens with the same, normalized word.
# Example: "ran," "runs," and "running" will all be converted to "run"
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))

    return lemmatized_sentence


# Removes many of the unnecessary words and punctuation in the token list
# Also removes any of the words in the list of stop_words which aren't important to the context of the word
def remove_noise(tokens, stop_words_list=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' 
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words_list \
                and token != '’' and token != "'s" and token != "n't" and token != "''" \
                and token != "``" and token != "”" and token != "..." and token != "“":
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


def load_training_data():
    print("Loading in training data...")
    positive_data = open('positive_data.json')
    negative_data = open('negative_data.json')
    return json.load(positive_data), json.load(negative_data), stopwords.words('english')


def clean_dataset(positive_post_list, negative_post_list, stop_words_list):
    print("Cleaning dataset...")
    # Tokenize the titles into a list of words rather than a string that is a sentence
    positive_tokenized_titles = []
    negative_tokenized_titles = []

    for post in positive_post_list:
        positive_tokenized_titles.append(nl.word_tokenize(post['title']))

    for post in negative_post_list:
        negative_tokenized_titles.append(nl.word_tokenize(post['title']))

    # Clean the token list by lemmatizing (normalizing) the tokens and removing unnecessary words and punctuation
    positive_clean_tokens = []
    negative_clean_tokens = []

    for title in positive_tokenized_titles:
        positive_clean_tokens.append(remove_noise(lemmatize_sentence(title), stop_words_list))

    for title in negative_tokenized_titles:
        negative_clean_tokens.append(remove_noise(lemmatize_sentence(title), stop_words_list))

    # Turn the list of tokens into a dictionary that associates the list of tokens with either "Positive" or "Negative"
    positive_dataset_res = [(title_dict, "Positive")
                            for title_dict in positive_clean_tokens]

    negative_dataset_res = [(title_dict, "Negative")
                            for title_dict in negative_clean_tokens]

    return positive_dataset_res, negative_dataset_res


def train_and_test_model(full_dataset, model_classifier):
    # Split the dataset roughly 70/30 into training and testing data
    random.shuffle(full_dataset)
    train_data = full_dataset[:700]
    test_data = full_dataset[700:]

    # Gets a list of all the words used in the dataset
    all_words = set(word.lower() for passage in train_data for word in passage[0])

    # Convert tokens into a form acceptable to the classifier
    t = [({word: (word in x[0]) for word in all_words}, x[1]) for x in train_data]

    # Perform the training on a simple naive Bayesian classifier
    print("Training the model...")
    model_classifier = NaiveBayesClassifier.train(t)

    # Calculate the accuracy of the model on the testing dataset
    print("Calculating accuracy of model on validation data...")
    tmp = [({word: (word in x[0]) for word in all_words}, x[1]) for x in test_data]
    print("Accuracy is:", classify.accuracy(model_classifier, tmp))

    # Prints the words that had the most effect on the model's decisions
    print(model_classifier.show_most_informative_features(10))

    return model_classifier


def get_sentiment_data(model_classifier, num_weeks):
    print("Getting sentiment data...")
    start_day = dt.datetime.today() - dt.timedelta(days=1)

    f = open('recent_data.json')
    recent_data = json.load(f)

    x_data = []
    y_data = []

    for i in range(num_weeks):
        date = (start_day - dt.timedelta(days=(i * 7)))
        date_str = date.strftime("%m/%d")
        x_data.append(date_str)

    print("Classifying titles...")

    for week in recent_data:
        positives = 0
        negatives = 0
        for json_title in week:
            clean_tokens = remove_noise(lemmatize_sentence(nl.word_tokenize(json_title['title'])))
            res = model_classifier.classify(dict([token, True] for token in clean_tokens))
            if res == "Positive":
                positives = positives + 1
            else:
                negatives = negatives + 1
        y_data.append(positives / (positives + negatives))

    # The data needs to be reversed so that it is displayed in chronological order
    x_data.reverse()
    y_data.reverse()

    return x_data, y_data


def plot_sentiment(x_data, y_data):
    # Plotting the sentiment score week-by-week over the time period specified in cmd line arg
    print("Plotting data...")
    plt.plot(x_data, y_data, color='red', marker='o')
    plt.title("Percentage of post titles from r/cryptocurrency deemed 'positive' over time")
    plt.xlabel("Date")
    plt.ylabel("Percentage of titles deemed positive")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    positive_posts, negative_posts, stop_words = load_training_data()

    positive_dataset, negative_dataset = clean_dataset(positive_posts, negative_posts, stop_words)
    dataset = positive_dataset + negative_dataset

    classifier = None
    classifier = train_and_test_model(dataset, classifier)

    graph_data_crawler.get_titles_from_past_weeks(int(sys.argv[1]))

    sentiment_x_data, sentiment_y_data = get_sentiment_data(classifier, int(sys.argv[1]))

    plot_sentiment(sentiment_x_data, sentiment_y_data)

    # Additional, informational graphs plotted to aid in analyzing the usefulness of the model

    extra_graphs.get_btc_price_graph()

    btc_gain_ydata = extra_graphs.get_btc_price_percent_graph()

    extra_graphs.get_sentiment_to_btc_percent_diff_graph(sentiment_x_data, sentiment_y_data, btc_gain_ydata)
