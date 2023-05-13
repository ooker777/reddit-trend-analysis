import praw
import pandas as pd
import numpy as np
from datetime import datetime

# create a reddit connection
reddit = praw.Reddit(
    client_id="ZjuiHwUJu2nxcHRb6CdK8g",
    client_secret="27eTTpL9IcjVgM5ca8zldhFIkHBO6A", 
    user_agent="testscript by u/ooker777"
)

i = 1
for submission in reddit.subreddit("all").hot(limit=50):
    print(i, " ", submission.title)
    i+=1

# posts = []
# for post in reddit.subreddit("all").hot(limit=50):

#     posts.append(
#         [
#             post.title,
#             post.author_flair_text,
#             # post.id,
#             # post.subreddit,
#             # post.url,
#             # post.num_comments,
#             # post.selftext,
#             # post.created,
#         ]
#     )
# posts = pd.DataFrame(
#     posts,
#     columns=[
#         "title",
#         "score",
#         # "id",
#         # "subreddit",
#         # "url",
#         # "num_comments",
#         # "body",
#         # "created",
#     ],
# )
# # posts["created"] = pd.to_datetime(posts["created"], unit="s")
# print(posts.head())
# posts