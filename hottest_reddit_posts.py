# get the start time
import time
st = time.time()

from langdetect import detect
import datetime
import praw
import pandas as pd
import os

# create a reddit connection
reddit = praw.Reddit(
    client_id="ZjuiHwUJu2nxcHRb6CdK8g",
    client_secret="27eTTpL9IcjVgM5ca8zldhFIkHBO6A", 
    user_agent="testscript by u/ooker777"
)
def get_data(subs, limit):
    queried_result = reddit.subreddit(subs).hot(limit=limit)
    posts = []
    i = 1
    for post in queried_result:
        if (
            # True
            len(post.selftext) > 550          # Độ dài bài viết > 550 ký tự
            and len(post.title) > 15          # Độ dài tiêu đề > 15 ký tự
            and len(post.title) < 345         # Độ dài tiêu đề < 345 ký tự
            and post.selftext != "[removed]"  # Bài viết không bị xoá
            and post.selftext != "[deleted]"  # Bài viết không bị xoá
            and detect(post.title) == 'en'    # Tiêu đề viết bằng tiếng Anh
        ):
            print(i, post.title, post.subreddit)
            i+=1
            posts.append([
                post.created_utc,
                post.subreddit,
                post.author,
                post.domain,
                post.url,
                post.num_comments,
                post.score,
                post.title,
                post.selftext,
                post.id,
                post.gilded,
                # post.retrieved_on,
                post.over_18,
            ])
    posts = pd.DataFrame(
        posts,
        columns=[
            "created_utc",
            "subreddit",
            "author",
            "domain",
            "url",
            "num_comments",
            "score",
            "title",
            "selftext",
            "id",
            "gilded",
            # "retrieved_on",
            "over_18",
        ],
    )
    # posts["created"] = pd.to_datetime(posts["created"], unit="s")
    # posts.to_excel('Reddit posts.xlsx', sheet_name='new_sheet_name')
    # print(posts.head())
    # os.startfile('Reddit posts.xlsx')
    pickle_path=f'data/{subs}{len(posts)}.pkl'
    posts.to_pickle(pickle_path)

    return len(posts)

subs="all" 
limit=50000
total_entries = get_data(subs, limit) 
print("Subreddit:", subs)
print("Limit:", limit)
print("CWD:", os.getcwd())
print("Reddit user:", reddit.user.me())

# logging
with open("log.md", "a+") as file_object:
    file_object.write(f'{datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")}\t{subs}\t{limit}\t{total_entries}\n')
et = time.time()
elapsed_time = (et - st)/60
print('Execution time:', elapsed_time, 'minutes')