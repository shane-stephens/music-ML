# I'm particularly proud of the final classifier accuracy of this project (74%), considering I have 4 groups. I set about trying to use machine learning to 
# define the abstract "vibe" that I felt when I listened to music. I had 4 distinct playlists that each had songs spanning across the same genres. 
# They were: music I would study to, music I would listen to at night, music I would listen to with my windows down in the car, and music I would work out to. 
# Since these playlists all included hip-hop, rock, and R&B songs, the algorithm wasn't merely splitting based on genre; I was trying to pick up on the features
# that my brain was subconciously using to categorize these songs. Would my algorithm be able to place songs into the correct buckets?
# While the sample size is small, I'm continuously adding to the playlists and re- running my code, honing the algorithm in on the specific features that 
# my brain is processing. 


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time 
from pandasql import sqldf


client_id = '' #insert client id
client_secret = '' # insert client secret id

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#Getting the tracks/features for each playlist

def getTrackIDs(user, playlist_id):
    ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids

#Since I was experimenting with different playlists, I just downloaded each one seperately. There's a faster way to do this, but I was lazy. The next lines
# are repeating code to download each playlist and put it together in a dataframe. Line #210 is where the ML starts.

#windows

ids = getTrackIDs('', '') #username, then playlist ID

def getTrackFeatures(id):
  meta = sp.track(id)
  features = sp.audio_features(id)

  # meta
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']
  valence = features[0]['valence']

  track = [name, album, artist, release_date, length, popularity, danceability, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence]
  return track

# loop over track ids 
tracks = []
for i in range(len(ids)):
  time.sleep(.5)
  track = getTrackFeatures(ids[i])
  tracks.append(track)

# create dataset
windows = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
windows['playlist'] = 'windows_down'

#night

ids = getTrackIDs('', '')

def getTrackFeatures(id):
  meta = sp.track(id)
  features = sp.audio_features(id)

  # meta
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']
  valence = features[0]['valence']

  track = [name, album, artist, release_date, length, popularity, danceability, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence]
  return track

# loop over track ids 
tracks = []
for i in range(len(ids)):
  time.sleep(.5)
  track = getTrackFeatures(ids[i])
  tracks.append(track)

# create dataset
night = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
night['playlist'] = 'night'

#Study

ids = getTrackIDs('', '')

def getTrackFeatures(id):
  meta = sp.track(id)
  features = sp.audio_features(id)

  # meta
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']
  valence = features[0]['valence']

  track = [name, album, artist, release_date, length, popularity, danceability, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence]
  return track

# loop over track ids 
tracks = []
for i in range(len(ids)):
  time.sleep(.5)
  track = getTrackFeatures(ids[i])
  tracks.append(track)

# create dataset
study = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
study['playlist'] = 'study'

#hoop

ids = getTrackIDs('', '')

def getTrackFeatures(id):
  meta = sp.track(id)
  features = sp.audio_features(id)

  # meta
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']
  valence = features[0]['valence']

  track = [name, album, artist, release_date, length, popularity, danceability, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence]
  return track

# loop over track ids 
tracks = []
for i in range(len(ids)):
  time.sleep(.5)
  track = getTrackFeatures(ids[i])
  tracks.append(track)

# create dataset
hoop = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'])
hoop['playlist'] = 'hoop'


all = pd.concat([windows,hoop,study,night])

music_feature=all[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'length']]


# NOW WE START THE MACHINE LEARNING!

#Split
from sklearn.model_selection import train_test_split
X = all[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'length']]
Y = all[['playlist']]
X_train, X_final, Y_train, Y_final = train_test_split(X,Y, test_size=.3, random_state=37, stratify=Y)

#CLASSIFYING TIME!

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#random state of 37 in honor of my little league career
classifier = RandomForestClassifier(n_estimators = 20, random_state=37)
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_final)

print('Accuracy:', metrics.accuracy_score(Y_final, y_pred))

#hyperparameter tuning 

param={'max_depth': [5,7,9], 'criterion': ['gini', 'entropy']}

from sklearn.model_selection import GridSearchCV
classifier=RandomForestClassifier(random_state=0)

Grid_search=GridSearchCV(classifier, param, cv=3)
Grid_search.fit(X_train,Y_train)
Grid_search.best_params_

#Final 
#Random state of 55 to commemorate the beautiful interstate I live off of 

final = RandomForestClassifier(n_estimators = 200, random_state=55, max_depth=9, criterion='gini')
final.fit(X_train,Y_train)
y_pred=final.predict(X_final)

print('Accuracy:', metrics.accuracy_score(Y_final, y_pred))
#.74--- WOO


estimator = final.estimators_[5]

#Visualizing the features in my decision tree. One of the perks of a decision tree is its relative interpretability compared to other ML algorithms. 

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, 
                rounded = True, proportion = False, 
                precision = 2, filled = True)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

tree.plot_tree(classifier);














#PCA

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
music_feature.loc[:]=min_max_scaler.fit_transform(music_feature.loc[:])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(music_feature)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

import matplotlib.pyplot as plt
all.reset_index(inplace=True)

principalDf['playlist']=all['playlist']
principalDf['name']=all['name']

import seaborn as sns
g =sns.scatterplot(x="principal component 1", y="principal component 2",
              hue="playlist",
              data=principalDf);




