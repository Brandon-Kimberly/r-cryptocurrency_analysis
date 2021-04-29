import json
import praw


# This function grabs (nearly) the last 1000 posts on the cryptocurrency subreddit. After obtaining them,
# it prints the titles of each one sequentially and waits for input from the user to determine if the sentiment of
# the title is positive or negative (in regards to the cryptocurrency subreddit).
def classify_last_thousand_posts():
    print("Connecting to Reddit...")
    reddit = praw.Reddit(client_id='', client_secret='',
                         user_agent='Crypto_Crawler')

    reddit.config.log_requests = 1
    reddit.config.store_json_result = True

    posts = []
    fields = ('title', 'score', 'url', 'selftext')

    cr_subreddit = reddit.subreddit('Cryptocurrency')

    print("Crawling through posts...")
    for post in cr_subreddit.new(limit=1000):
        to_dict = vars(post)
        sub_dict = {field: to_dict[field] for field in fields}
        posts.append(sub_dict)

    positive_titles = []
    negative_titles = []

    print("Classify the following submission titles: ")
    for post in posts:
        print(post['title'])
        choice = input("Type 1 for positive, anything else for negative, then hit enter: ")
        if str(choice) == "1":
            print("Positive")
            positive_titles.append(post)
        else:
            print("Negative")
            negative_titles.append(post)

    with open('positive_data.json', 'w') as f:
        json.dump(positive_titles, f)

    with open('negative_data.json', 'w') as f:
        json.dump(negative_titles, f)
