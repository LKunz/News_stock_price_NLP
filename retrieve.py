###############################################################################
#                    ADVANCED DATA ANALYSIS 2019 - PROJECT
###############################################################################

# This code retrieves the data from Reuters eikon 
# For data analysis, please see data_treatment.py

# Import packages and define eikon key
import eikon as ek 
import pandas as pd 
import datetime 
import warnings 
warnings.filterwarnings('ignore')
ek.set_app_key('PLEASE INSERT EIKON KEY')

# Number of days from today you want the news
today = datetime.datetime.today()
numdays = 450

# Create list of all dates
datelist = []
for t in range(0, numdays):
    datelist.append((today - datetime.timedelta(days = t)).date())
        
# Retrieve daily headlines
APIcode = 'R:AAPL.O AND Language:LEN'
frames = []
for date in datelist:
    # Volontary slow down request (avoid too many requests error from eikon)
    for i in range(0,1000):
        print('I love data science')
    # Get headline
    df = ek.get_news_headlines(APIcode, date_from=date, date_to=date, count=100) 
    frames.append(df)

# Create news dataframe
news = pd.concat(frames)

# Drop duplicates
news = news.drop_duplicates(subset='text', keep='last')

# Retrieve content of news
story = []
for storyId in news['storyId'].values:
    # Volontary slow down request (avoid too many requests error from eikon)
    for i in range(0,1000):
        print('I love data science')
    # Get news content
    try:
        newsText = ek.get_news_story(storyId)
        if newsText:
            story.append(newsText)
        else:
                story.append('')
    except:
        story.append('Error')
        
news['story'] = story

# Export data in csv format
news.to_csv('News.csv')



