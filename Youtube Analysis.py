#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Importing the Libraries

import pandas as pd
import numpy as np
#from googleapiclient.discovery import build
import os
#from googleapiclient.errors import HttpError
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import regex as re


# In[8]:


api_key = '########'


# In[21]:


vdolinks = pd.read_csv('vdoLinks.csv')
vdolinks.head()


# In[22]:


links = list(vdolinks['youtubeId'])


# In[ ]:


youtube_object = build("youtube", "v3", developerKey = api_key)


# In[ ]:


youtube_videodata = pd.DataFrame(columns=['title','description','view_count','like_count','dislike_count','comment_count',
                                'duration','favorite_count'])
for link in links:
    video_data = {}
    video_details = youtube_object.videos().list(part='snippet, statistics, contentDetails', id=link).execute()
    
    if len(video_details['items']) != 0:
        title = video_details['items'][0]['snippet'].get('title', 'NULL')
        video_data['title'] = title

        description = video_details['items'][0]['snippet'].get('description', 'NULL')
        video_data['description'] = description

        view_count = video_details['items'][0]['statistics'].get('viewCount', 0)
        video_data['view_count'] = view_count

        like_count = video_details['items'][0]['statistics'].get('likeCount', 0)
        video_data['like_count'] = like_count

        dislike_count = video_details['items'][0]['statistics'].get('dislikeCount', 0)
        video_data['dislike_count'] = dislike_count

        comment_count = video_details['items'][0]['statistics'].get('commentCount', 0)
        video_data['comment_count'] = comment_count

        duration = video_details['items'][0]['contentDetails'].get('duration', 0)
        video_data['duration'] = duration

        favorite_count = video_details['items'][0]['statistics'].get('favoriteCount', 0)
        video_data['favorite_count'] = favorite_count

        youtube_videodata = youtube_videodata.append(video_data, ignore_index=True)
youtube_videodata.head()


# In[ ]:


youtube_videodata.to_csv('youtube_videodata.csv')


# In[ ]:


youtube_commentsdata = pd.DataFrame(columns=['comments', 'link'])

for link in links:
    comments_data = {}
    try:
        video_comments = []
        comments_data['link'] = link
        comments = youtube_object.commentThreads().list(part='snippet', videoId=link, maxResults=100).execute()
        for comment in comments['items']:
            video_comments.append(comment['snippet']['topLevelComment']['snippet']['textDisplay'])
        comments_data['comments'] = video_comments
        youtube_commentsdata = df_comments.append(comments_data, ignore_index=True)
        
    except HttpError as error:
            if error.resp.status == 404:
                pass
youtube_commentsdata.head()


# In[ ]:


youtube_commentsdata.to_csv('youtube_commentsdata.csv')


# ## Top 10 videos based on view count

# In[23]:


youtube_videodata = pd.read_csv('youtube_videodata.csv', usecols=range(1,9))
youtube_videodata.head()


# In[24]:


youtube_videodata_sorted = youtube_videodata.sort_values(by=['view_count'], ascending=False)
youtube_videodata_sorted[:10]


# In[25]:


youtube_videodata_sorted[['title', 'view_count']][:10]


# In[26]:


fig = plt.figure(figsize = (8, 5))
plt.bar(youtube_videodata_sorted[:10]['title'], youtube_videodata_sorted[:10]['view_count'])
plt.xlabel("Movie names")
plt.ylabel("Views")
plt.xticks(rotation= 90)
plt.title("Top 10 videos with total views")
plt.show()


# ### Bottom 10 videos based on view count

# In[27]:


youtube_videodata_sorted[-10:]


# In[28]:


bottom_videos = youtube_videodata_sorted[(youtube_videodata_sorted['view_count'] == 1) | (youtube_videodata_sorted['view_count'] == 2)][-10:]
bottom_videos


# In[29]:


fig = plt.figure(figsize = (8, 5))
plt.bar(bottom_videos['title'], bottom_videos['view_count'])
plt.xlabel("Movie names")
plt.ylabel("Views")
plt.xticks(rotation= 90)
plt.title("Bottom 10 videos with total views")
plt.show()


# ### Most liked video

# In[30]:


mostliked_video = youtube_videodata[youtube_videodata['like_count'] == youtube_videodata['like_count'].max()]
mostliked_video


# ### Least liked video

# In[31]:


least_liked_video = youtube_videodata[(youtube_videodata['like_count'] == 0)]
least_liked_video


# In[32]:


least_liked_video = youtube_videodata[(youtube_videodata['like_count'] == 1)]
least_liked_video


# In[33]:


least_liked_video.shape


# ### Video with highest duration

# In[36]:


import isodate

def duration_to_seconds(df):
    dur = isodate.parse_duration(df)
    return (dur.total_seconds())


# In[37]:


#converting duration to seconds
youtube_videodata['duration'] = youtube_videodata['duration'].apply(duration_to_seconds)
youtube_videodata.head()


# In[38]:


high_duration_video = youtube_videodata[youtube_videodata['duration'] == youtube_videodata['duration'].max()]
high_duration_video


# ### Sentiment Analysis

# In[15]:


youtube_commentsdata = pd.read_csv('youtube_commentsdata.csv', usecols=range(1,3))


# In[16]:


vdolinks = pd.read_csv('vdoLinks.csv')
vdolinks.rename(columns={'youtubeId':'link'}, inplace=True)
vdolinks.head()


# In[17]:


import numpy as np
youtube_commentsdata = pd.merge(youtube_commentsdata, vdolinks[['link', 'title']], on='link', how='left' )
youtube_commentsdata.head()


# In[19]:


youtube_commentsdata['comments'][0]


# In[28]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from bs4 import BeautifulSoup
import unicodedata


# In[29]:


analyser = SentimentIntensityAnalyzer()


# In[30]:


sentiment_score = pd.DataFrame(columns=['title', 'scores', 'sentiment'])
for i in range(youtube_commentsdata.shape[0]):
    comments = re.findall("\'([^\\']*)\'", youtube_commentsdata['comments'][i])
    if len(comments)>0:
        scores = []
        title = youtube_commentsdata['title'][i]
        for text in comments:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
            text = re.sub(r'[0-9]', ' ', text)
            text = BeautifulSoup(text, 'html.parser').get_text()
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            score = analyser.polarity_scores(text)
            scores.append(score['compound'])
        sentiment = sum(scores) / len(scores)
        sentiment_score = pd.concat([sentiment_score, pd.DataFrame.from_records([{'title':title, 'scores':scores, 'sentiment':sentiment }])], ignore_index=True)
    
sentiment_score.head()


# In[34]:


sentiment_score = sentiment_score.sort_values(by=['sentiment'], ascending=False)
fig = plt.figure(figsize = (8, 5))
plt.bar(sentiment_score['title'][:10], sentiment_score['sentiment'][:10]) 
plt.xlabel("Movie names")
plt.ylabel("positive sentiment")
plt.xticks(rotation= 90)
plt.title("Top 10 videos that have positive sentiments")
plt.show()


# In[ ]:




