
import os
import sys
from io import open


# for data and saves
import pandas as pd
import numpy as np
import dill
from PIL import Image # pillow package


# custom package
from emlyon_module.structured import *


# for app
import streamlit as st



#**********************************************************
#*                      functions                         *
#**********************************************************

# blogs on this topic
# https://blog.jcharistech.com/2019/11/28/summarizer-and-named-entity-checker-app-with-streamlit-and-spacy/
# https://blog.jcharistech.com/2019/12/14/building-a-document-redactor-nlp-app-with-streamlitspacy-and-python/

# ------------------------- Paths -------------------------
path_to_rep  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_data = os.path.join(path_to_rep, 'data', "archive", 'genre_V3.csv')
path_to_raw_data = os.path.join(path_to_rep, 'data', "archive", 'genres_v2.csv')


# ------------------------- Utils -------------------------
from PatchStreamlit import (
    SessionState, 
    st_rerun,
)


# ------------------------- Layout ------------------------
def centerText(text, thick = 1) :
    '''Displays a text with centered indentation, with specified thickness (the lower, the thickier)'''
    st.markdown("<h{} style='text-align: center; color: black;'>{}</h{}>".format(thick, text, thick), unsafe_allow_html = True)
    return



# -------------------------- Tmp --------------------------

def display_genre(index):
    dict_genre = {7: 'Dark Trap',
 12: 'Emo',
 11: 'Hiphop',
 10: 'Pop',
 9: 'Rap',
 1: 'RnB',
 13: 'Trap Metal',
 6: 'Underground Rap',
 5: 'dnb',
 14: 'hardstyle',
 8: 'psytrance',
 4: 'techhouse',
 0: 'techno',
 2: 'trance',
 3: 'trap'}

    # compute model prediction
    pred_genre = model.predict(X.values)[index]
    true_genre = y[index]

    # display actual and predicted prices
    col_valid, col_pred = st.beta_columns(2)
    with col_valid:
        centerText('real genre', thick = 3)
        centerText(str(dict_genre[true_genre]), thick = 4)
    with col_pred:
        centerText('estimated genre', thick = 3)
        centerText(str(dict_genre[pred_genre]), thick = 4)
    return


def display_features(index):
    centerText('Track features', thick = 3)
    feat0, val0, feat1, val1 = st.beta_columns([3.5, 1.5, 3.5, 1.5])
    row = X.values[index]
    for i, feature in enumerate(X.columns):
        ind = i % 2
        if ind == 0:
            with feat0:
                st.warning(feature)
            with val0:
                st.info(str(row[i]))
        elif ind == 1:
            with feat1:
                st.warning(feature)
            with val1:
                st.info(str(row[i]))
    return



#**********************************************************
#                     main script                         *
#**********************************************************


# session state
session_state = SessionState.get(
    model = None,
    X = None,
    y = None,
    imgs = None,
)


# init session state
if session_state.model is None:
    # validation set given in notebook
    n_valid = 20000

    # load and preprocess data
    data = pd.read_csv(
        path_to_data, 
        low_memory = False,
    )
    
    data_raw = pd.read_csv(path_to_raw_data,low_memory = False)
    song_name = data_raw.song_name

    X, y, nas = proc_df(data, 'genre')
    X, y = X[n_valid:], y[n_valid:]

    # load regression model
    path_to_model = os.path.join(path_to_rep, 'app Streamlit', 'saves', 'RF_classifier.pk')
    with open(path_to_model, 'rb') as file:
        model = dill.load(file)

centerText('Choose a song', thick = 1)
st.write(' ')
st.write(' ')

index = st.selectbox(
    'select a song', 
    options = ['-'] + [i for i in range(1, n_valid + 1)],
    index = 0,
)


if type(index) == int:
    
    centerText("Song Name:" + song_name[index])
    # track genre
    centerText('Genre Prediction', thick = 3)
    void0, col, void1 = st.beta_columns([4, 2, 4])
    with col:
        estimate = st.button('Estimate !')
    if estimate:
        display_genre(index)

    # track features
    display_features(index)

