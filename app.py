import time
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import food_recomm
import recipe_finder

st.set_page_config(
    page_title="Mommy's Food",
    page_icon=':pizza:'
)

@st.cache()
def fetch_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    for i in range(0,len(df["Ingredients"])):
        df["Ingredients"][i] =  re.sub(r'\'','',df["Ingredients"][i])

    # get all items
    items = set()
    for x in df.Ingredients:
        for val in x.split(', '):
            items.add(val.lower().strip())
        # break
    # items = sorted(items)

    # create new dataframe
    new_df = pd.DataFrame(data=np.zeros((len(df), len(items)+5), dtype=int), columns=['Dish', 'Ingredients','URL','Class','Cuisine'] + list(items))


    for i, d in df.iterrows():
        new_df.loc[i, ['Dish', 'Ingredients','URL','Class','Cuisine']] = d[:5]

        for val in d[1].split(', '):
            item = val.lower().strip()
            new_df.loc[i, item] = 1

    return new_df


data = fetch_and_clean_data('data/Final_Data.csv')

PAGES = {
    'Recipe Finder': recipe_finder,
    'Food Recommender': food_recomm
}

page = st.sidebar.radio(
    label='Contents',
    options=list(PAGES.keys())
)




content = PAGES[page]
content.app(data)
