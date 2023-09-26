import streamlit as st
import torch
import filelock
from transformers import AutoModelForSequenceClassification
from newsapi import NewsApiClient
# from newscatcherapi import NewsCatcherApiClient
import csv
import pandas as pd
from streamlit import session_state

# st.session_state[TABLE_STATE] = False
st.set_page_config(page_title='sentiment Tool', page_icon='', initial_sidebar_state='expanded')
st.sidebar.header("Sentiment Tools `Version 1 by Tamfu`")
# API client
newsapi = NewsApiClient(api_key='beab2130cda447d4b451d64146ea0780')
model_name = "FinanceInc/auditor_sentiment_finetuned" 
model =AutoModelForSequenceClassification.from_pretrained(model_name)

# # def fetch_articles(keyword,related_topic,):
# #     try:
# #         all_articles = newscatcherapi.get_search(q= keyword,
# #                                             lang='en',
# #                                             topic= related_topic,
# #                                             from_ = '1 days ago',
# #                                             from_rank= 1,
# #                                             #countries= 'US',
# #                                             to_rank= 1000,
# #                                             sort_by= 'relevancy',
# #                                             page_size=100)    
#     except Exception as e:
#        print(e)
#        return None

#     if all_articles['total_hits'] == 0:
#         return None

#     return all_articles
def fetch_articles(keyword,related_topic,country_slt):
    try:
        all_articles = newsapi.get_top_headlines(q= keyword,
                                          category= related_topic,
                                          language='en',
                                          country=country_slt)    
    except Exception as e:
       print(e)
       return None

    if all_articles['totalResults'] == 0:
        return None

    return all_articles   


def extract_titles(articles):
    if articles is None:
        return []

    titles = [a['title'] for a in articles['articles']]  
    return titles

# def load_model(titles):
#     from transformers import pipeline
#     stm_pipeline = pipeline("sentiment-analysis", model=model_name, framework='pt')
#     return stm_pipeline(titles)

def analyze_sentiment(titles):
    from transformers import pipeline
    stm_pipeline = pipeline("sentiment-analysis", model=model_name, framework='pt')
    if not titles:
        return "No titles"

    positive = 0
    negative = 0
    neutral = 0

    predictions = stm_pipeline(titles)

    for prediction in predictions:
        if prediction['label'] == 'POSITIVE':
            positive += prediction['score']
        elif prediction['label'] == 'NEGATIVE': 
            negative += prediction['score']
        else:
            neutral += prediction['score']

    total_titles = len(titles)
    pos_avg = positive / total_titles
    neg_avg = negative / total_titles
    neu_avg = neutral / total_titles

    if pos_avg > neg_avg and pos_avg > neu_avg:
        return 'positive'
    elif neg_avg > pos_avg and neg_avg > neu_avg:  
        return 'negative'
    else:
        return 'neutral'

# Création du formulaire
st.title("Formulaire d'analyse")
with st.form("my_form"):
    # from pages.alert import verif_topic
    keyword = st.text_input("Actif", placeholder="entrez l'actif",
                            help="une phrase ou un mot").lower()
    topic = st.text_input("cathégorie", placeholder="entrez le thème",
                          help= 'la cathégorie doit etre comprise dans la liste suivante ("entertainment","general", "health","science","sport","technology","business")').lower()
    country = st.selectbox('Pays', ['us','ru','gb','ca','ru','ch','ua',"fr"])

    # verif_topic(topic)
    submit_button = st.form_submit_button("Analyser")

# Vérification si le bouton "Analyser" a été cliqué
sentiment=None
# valid_topics = ['news', 'sport', 'tech', 'world', 'finance', 'politics', 
#                 'business', 'economics', 'entertainment', 'beauty', 
#                 'travel', 'music', 'food', 'science', 'gaming', 'energy']
valid_topics = ["entertainment","general", "health","science","sport","technology","business"]  

# def download_csv(titles):
#   with open('titles.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Title'])
#     writer.writerows([[t] for t in titles])
  
#   with open('titles.csv', 'rb') as f:
#     btn = st.download_button(label='Download CSV', data=f, file_name='titles.csv')

def download_titles(titles):

  df = pd.DataFrame(titles, columns=['Title'])
  
  df.to_csv('titles.csv', index=False)

  with open('titles.csv') as f:
    st.download_button('Télécharger les titres', f, file_name='titles.csv')

if submit_button:
    if topic not in valid_topics:
        st.error("Thème invalide")
    else:
        with st.spinner("Loading..."):
                articles = fetch_articles(keyword, topic,country)
                titles = extract_titles(articles)
                sentiment = analyze_sentiment(titles)

                if sentiment == "No titles":
                    st.warning("No articles found")
                else:
                    df = pd.DataFrame(columns=['Title']) 
                    st.success(f"{sentiment}")
                    # st.button("Download Titles", 
                    #           on_click=download_csv, 
                    #           args=(titles,)) 
                # if st.button("Download Titles") :

                download_titles(titles)   



 

