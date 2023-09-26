#Headlines-sentiment-tool

this tool is a machine learning APP created with a  Transformers BERT pretrained model "FinanceInc/auditor_sentiment_finetuned" which analyze and give the overall sentiment 
over a list of headlines fetched through a request with the " NewsAPI ", the app contain 2 pages "Alert" & "Acceuil" Which represent respectively the alert page used to create an analysis alert whose result is sent via email entered in the form, and the homepage used to perform an instant analysis query whose result depends on the availability of the titles or not.

This application is still in the experimental stage.

#Installation 

this app is a personnal project, so you can just download the source code an run "streamlit acceuil.py" in the console

#Requirement

- streamlit
- python 3.10 +
- pytorch
- Transformers
- pandas
- and usual common libraries 
