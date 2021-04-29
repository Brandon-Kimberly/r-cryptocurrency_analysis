import praw
import json
import datetime as dt
from psaw import PushshiftAPI


# This function grabs a representative sample of submission titles from the r/cryptocurrency subreddit dating from
# today and going back __weeks__ weeks.
# NOTE: The "representative sample" is as follows; the first 400 posts from each day (according to UTC) in the week
#       are combined into one list of length 7*400=2800 submissions that represents the week.
# NOTE 2: This function will take a while because of the rate limiting done by Reddit with their API.
def get_titles_from_past_weeks(weeks):
    print("Gathering data from past weeks to plot the sentiment score...")
    number_of_weeks = int(weeks)

    reddit = praw.Reddit(client_id='', client_secret='',
                         user_agent='Crypto_Crawler')
    r_api = PushshiftAPI(reddit)

    start_day = dt.datetime.today() - dt.timedelta(days=7)
    end_day = dt.datetime.today()
    start_epoch = int(dt.datetime(start_day.year, start_day.month, start_day.day).timestamp())
    end_epoch = int(dt.datetime(end_day.year, end_day.month, end_day.day).timestamp())

    res = []

    for i in range(number_of_weeks):
        print("Gathering data for week " + str(i + 1) + "...")
        week = []
        for j in range(7):
            day = []
            for submission in r_api.search_submissions(after=start_epoch, before=start_epoch + (60*60*24*j),
                                                       subreddit='cryptocurrency', filter=['title'],
                                                       limit=400):
                to_dict = vars(submission)
                sub_dict = {'title': to_dict['title']}
                day.append(sub_dict)
            week = week + day
        start_epoch = start_epoch - (60 * 60 * 24 * 7)
        res.append(week)

    print("Dumping data to JSON file...")
    with open('recent_data.json', 'w') as f:
        json.dump(res, f)
