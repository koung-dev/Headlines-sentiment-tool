import streamlit as st
import torch
import filelock
from transformers import AutoModelForSequenceClassification
from newscatcherapi import NewsCatcherApiClient
from newscatcherapi import NewsCatcherApiClient
import csv
import pandas as pd

st.set_page_config(page_title='sentiment Tool', page_icon='', initial_sidebar_state='expanded')
st.sidebar.header("Sentiment Tools `Version 1 by Tamfu`")
selected_option = st.sidebar.selectbox("Historique", ["Alerte", "Analyse"])
# API client
newscatcherapi = NewsCatcherApiClient(x_api_key="0xceRX_vZJ7YKhejGzcNx8o0psN7ppd6MOYcasn8C0c")
model_name = "FinanceInc/auditor_sentiment_finetuned" 
model =AutoModelForSequenceClassification.from_pretrained(model_name)

def fetch_articles(keyword,related_topic,):
    try:
        all_articles = newscatcherapi.get_search(q= keyword,
                                            lang='en',
                                            topic= related_topic,
                                            from_ = '1 days ago',
                                            from_rank= 1,
                                            #countries= 'US',
                                            to_rank= 1000,
                                            sort_by= 'relevancy',
                                            page_size=100)    
    except Exception as e:
       print(e)
       return None

    if all_articles['total_hits'] == 0:
        return None

    return all_articles

def extract_titles(articles):
    if articles is None:
        return []

    titles = [a['title'] for a in articles['articles']]  
    return titles

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
    keyword = st.text_input("Actif", placeholder="entrez l'actif").lower()
    topic = st.text_input("Thème", placeholder="entrez le thème").lower()
    # verif_topic(topic)
    submit_button = st.form_submit_button("Analyser")

# Vérification si le bouton "Analyser" a été cliqué
sentiment=None
valid_topics = ['news', 'sport', 'tech', 'world', 'finance', 'politics', 
                'business', 'economics', 'entertainment', 'beauty', 
                'travel', 'music', 'food', 'science', 'gaming', 'energy']
if submit_button:
    if topic not in valid_topics:
        st.error("Thème invalide")
    else:
        with st.spinner("Loading..."):
                articles = fetch_articles(keyword, topic)
                titles = extract_titles(articles)
                sentiment = analyze_sentiment(titles)

                if sentiment == "No titles":
                    st.warning("No articles found")
                else:
                    st.success(f"{sentiment}")
                    # st.button("Download Titles", 
                    #           on_click=download_csv, 
                    #           args=(titles,)) 
                    st.button("Download Titles")
                    if st.button("Download Titles"):
                        df = pd.DataFrame(titles, columns=['Title'])
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name='titles.csv'
                        )    



 

