# -*- coding: utf-8 -*-
# This script is a way for me to generate my own personal discover weekly Spotify playlist based on a current playlist I already have. I also added the
# option for the user to input their current mood. For example, if I had a playlist of hip-hop songs I liked, and input that I was sad, the algorithm would
# create and generate within Spotify a playlist full of sad hip songs tailored to my needs. 

# The funcitonal way the algorithm works is as follows. First, I download all the song data from Spotify's API which includes abstract musical features such
# as "danceability", "valence", "acousticness", etc. These variables become my predictors, with the number of plays the user has on the song in the past year 
# being the predictor. I then pitted multiple machine learning algorithms against each other, with the algorithm that performed the "best" being the winner.
# From this, I get a playlist of equal size made up of all new and unique songs. 


#SENTIMENT ANALYSIS

from sentiment_module import sentiment
from textblob import TextBlob

valence = input("On a scale of 1-10 (10 being the happiest), I want a happiness level of: ")
print("Sentiment: ", valence)
energy = input("I want the energy level to be: ")
print("Energy level:", energy)

v_t= int(valence)/10
e_t= int(energy)/10

valence_min=v_t-.15
valence_max=v_t+.15
energy_min=e_t-.15
energy_max=e_t+.15


#CREATING THE BASIS PLAYLIST

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util


cid ='' # Client ID; copy this from your app created on beta.developer.spotify.com
secret = '' # Client Secret; copy this from your app
username = '' # Your Spotify username

scope = 'user-library-read playlist-modify-public playlist-read-private'

redirect_uri=''

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

#spotipy.oauth2.SpotifyOAuth(client_id=cid, client_secret=secret, redirect_uri=redirect_uri, state=None, scope=None, cache_path=None, username=username, proxies=None, show_dialog=False, requests_session=True, requests_timeout=None, open_browser=True)


import pandas as pd

sourcePlaylistID = '' #PUT THE PLAYLIST YOU WANT TO MIMIC HERE!
sourcePlaylist = sp.user_playlist(username, sourcePlaylistID);
tracks = sourcePlaylist["tracks"];
songs = tracks["items"];

track_ids = []
track_names = []

for i in range(0, len(songs)):
    if songs[i]['track']['id'] != None: # Removes the local tracks in your playlist if there is any
        track_ids.append(songs[i]['track']['id'])
        track_names.append(songs[i]['track']['name'])

features = []
for i in range(0,len(track_ids)):
    audio_features = sp.audio_features(track_ids[i])
    for track in audio_features:
        features.append(track)
        
playlist_df = pd.DataFrame(features, index = track_names)

#Did it work?
playlist_df.head(10)
#Yes

#Keeping just what we need
playlist_df=playlist_df[["id", "acousticness", "danceability", "duration_ms", 
                         "energy", "instrumentalness",  "key", "liveness",
                         "loudness", "mode", "speechiness", "tempo", "valence"]]
playlist_df.head()

#CREATING THE RANKINGS
import numpy as np
a_list = [10,10,10,10,10,10,10,10,10,10,20,20,20,20,20,20,20,20,20,20,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,40,40,40,40,40,40,50,50,50,50,50,50,50,50,50,50,60,60,60,60,60,60,60,60,60,60,70,70,70,70,70,70,70,70,70,70,80,80,80,80,80,80,80,80,80,80,90,90,90,90,90,90,90,90,90,90,100,100,100,100,100,100,100,100,100]

a_list.reverse()

playlist_df['ratings']=a_list

from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier

#Setting our predictors and targets
X_train = playlist_df.drop(['id', 'ratings'], axis=1)
y_train = playlist_df['ratings']
forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=12) # Set by GridSearchCV below
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature rankings, figure out what's important
print("Feature ranking:")
  
for f in range(len(importances)):
    print("%d. %s %f " % (f + 1, 
            X_train.columns[f], 
            importances[indices[f]]))
    
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')


X_scaled = StandardScaler().fit_transform(X_train)

#pca to condense the dataset
pca = decomposition.PCA().fit(X_scaled)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 12)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(9, c='b') # Tune this so that you obtain at least a 95% total variance explained
plt.axhline(0.95, c='r')
plt.show();

# Fit your dataset to the optimal pca- CHECK THIS INPUT!!!!!!
pca1 = decomposition.PCA(n_components=9)
X_pca = pca1.fit_transform(X_scaled)

from sklearn.manifold import TSNE

tsne = TSNE(random_state=17)
X_tsne = tsne.fit_transform(X_scaled)

from scipy.sparse import csr_matrix, hstack

#X_train_last = csr_matrix(hstack([X_pca, X_names_sparse])) # Check with X_tsne + X_names_sparse also
X_train_last=X_pca

from sklearn.model_selection import StratifiedKFold, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Initialize a stratified split for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Trees First
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}

tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)

tree_grid.fit(X_train_last, y_train)
tree_grid.best_estimator_, tree_grid.best_score_

#RANDOM FOREST
parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                             n_jobs=-1, oob_score=True)
gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv1.fit(X_train_last, y_train)
gcv1.best_estimator_, gcv1.best_score_

# kNN third
from sklearn.neighbors import KNeighborsClassifier

knn_params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier(n_jobs=-1)

knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
knn_grid.fit(X_train_last, y_train)
knn_grid.best_params_, knn_grid.best_score_

# Now build your test set;
# Generate a new dataframe for recommended tracks

rec_tracks = []
for i in playlist_df['id'].values.tolist():
    rec_tracks += sp.recommendations(seed_tracks=[i], limit=int(len(playlist_df)/2))['tracks'];

rec_track_ids = []
rec_track_names = []
for i in rec_tracks:
    rec_track_ids.append(i['id'])
    rec_track_names.append(i['name'])

rec_features = []
for i in range(0,len(rec_track_ids)):
    rec_audio_features = sp.audio_features(rec_track_ids[i])
    for track in rec_audio_features:
        rec_features.append(track)
        
rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)
rec_playlist_df.head()

#X_test_names = v.transform(rec_track_names)

rec_playlist_df=rec_playlist_df[["acousticness", "danceability", "duration_ms", 
                         "energy", "instrumentalness",  "key", "liveness",
                         "loudness", "mode", "speechiness", "tempo", "valence"]]
# Make predictions
tree_grid.best_estimator_.fit(X_train_last, y_train)
rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
rec_playlist_df_pca = pca1.transform(rec_playlist_df_scaled)
#X_test_last = csr_matrix(hstack([rec_playlist_df_pca, X_test_names]))
X_test_last = rec_playlist_df_pca
y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

rec_playlist_df['ratings']=y_pred_class
rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
rec_playlist_df = rec_playlist_df.reset_index()
rec_playlist_df = rec_playlist_df.drop_duplicates(subset = 'index')


#rec_playlist_df.hist(column='valence', bins=20, color="#86bf91")

# Pick the top ranking tracks to add your new playlist 9, 10 will work

rec_playlist_df = rec_playlist_df[(rec_playlist_df['valence'] >= valence_min) & (rec_playlist_df['valence'] <= valence_max)]
rec_playlist_df = rec_playlist_df[(rec_playlist_df['energy'] >= energy_min) & (rec_playlist_df['energy'] <= energy_max)]

recs_to_add=rec_playlist_df.head(len(songs))

recs_to_add = rec_playlist_df[rec_playlist_df['ratings']>=90]['index'].values.tolist()

#recs_to_add=rec_playlist_df.head(len(track_ids))


len(rec_tracks), rec_playlist_df.shape, len(recs_to_add)

rec_array = np.reshape(recs_to_add, (len(recs_to_add), 1))
rec_array = rec_array[:len(songs)]
#rec_array=recs_to_add
#rec_array=recs_to_add

#Create the playlist and write it into your account

playlist_recs = sp.user_playlist_create(username, 
                                        name='Indie Songs for Playlist - {}'.format(sourcePlaylist['name']))
for i in rec_array:
    sp.user_playlist_add_tracks(username, playlist_recs['id'], i);
    
#Done :)
    
    
