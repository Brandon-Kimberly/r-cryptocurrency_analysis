# r/Cryptocurrency Sentiment Analysis
The code (and example data) needed to:
  1. Download training data from Reddit's r/cryptocurrency subreddit.
  2. Create and train a naive Bayes classifier on the data.
  3. Graph the sentiment of the subreddit over time (and other informational graphs)


##  Description of each file

### main.py
This file is the main driver of the whole project. Executing just this file will run all the code that is contained in this repository. Therefore, the purpose of this file can be summarized in the same way as the top of the README.


### training_data_crawler.py
In this file, the Python Reddit API Wrapper (PRAW) service is used to gather the last one thousand posts on Reddit.com/r/cryptocurrency and dump it to a JSON file. This data will be utilized in main.py to train the probabilistic classifier.

NOTE: In order to use this file you will need to create an app [here](https://www.reddit.com/prefs/apps) and paste you client ID and secret in the proper place near the top of the file.

### graph_data_crawler.py
The purpose of this file is slightly different from training_data_crawler. This file utilizes a third-party extension of PRAW called PushshiftAPI. The service change is due to a restriction in the number of posts that can be obtained easily using PRAW. This file grathers a representative sample of submission titles from the present day, back a specified number of weeks.

NOTE: In order to use this file you will need to create an app [here](https://www.reddit.com/prefs/apps) and paste you client ID and secret in the proper place near the top of the file.

### extra_graphs.py
Here is some code to display additional interesting or informational graphs. This file is not necessary for training, creating, or using the model. It is simply used to aid in analysis of the usefulness of the model.
