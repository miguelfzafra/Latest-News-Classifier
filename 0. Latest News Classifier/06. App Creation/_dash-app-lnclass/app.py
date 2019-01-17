import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, Event
import plotly.graph_objs as go
import re

# Importing the  inputs

path_models = "Pickles/"
# SVM
path_svm = path_models + 'best_svc.pickle'
with open(path_svm, 'rb') as data:
    svc_model = pickle.load(data)

path_tfidf = "Pickles/tfidf.pickle"
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4,
    'other':5
}

# Definition of functions
# El Pais
def get_news_elpais():
    
    # url definition
    url = "https://elpais.com/elpais/inenglish.html"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html.parser')

    # News identification
    coverpage_news = soup1.find_all('h2', class_='articulo-titulo')
    len(coverpage_news)
    
    number_of_articles = 5

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # only news articles (there are also albums and other things)
        if "inenglish" not in coverpage_news[n].find('a')['href']:  
            continue

        # Getting the link of the article
        link = coverpage_news[n].find('a')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html.parser')
        body = soup_article.find_all('div', class_='articulo-cuerpo')
        x = body[0].find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame(
         {'Content': news_contents 
        })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Newspaper': 'El Pais English'})
    
    return (df_features, df_show_info)

# The Guardian
def get_news_theguardian():
    
    # url definition
    url = "https://www.theguardian.com/uk"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html.parser')

    # News identification
    coverpage_news = soup1.find_all('h3', class_='fc-item__title')
    len(coverpage_news)
    
    number_of_articles = 8

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # We need to ignore "live" pages since they are not articles
        if "live" in coverpage_news[n].find('a')['href']:  
            continue
            
        # We also need to ignore "commentisfree" pages
        if "commentisfree" in coverpage_news[n].find('a')['href']:  
            continue
            
        # And "ng-interactive" pages
        if "ng-interactive" in coverpage_news[n].find('a')['href']:  
            continue

        # Getting the link of the article
        link = coverpage_news[n].find('a')['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html.parser')
        body = soup_article.find_all('div', class_='content__article-body from-content-api js-article__body')
        x = body[0].find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame(
         {'Content': news_contents 
        })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Newspaper': 'The Guardian'})

    
    return (df_features, df_show_info)


# The Mirror
def get_news_themirror():
    
    # url definition
    url = "https://www.mirror.co.uk/"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html.parser')

    # News identification
    coverpage_news = soup1.find_all('a', class_='headline publication-font')
    len(coverpage_news)
    
    number_of_articles = 5

    # Empty lists for content, links and titles
    news_contents = []
    list_links = []
    list_titles = []

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = coverpage_news[n]['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html.parser')
        body = soup_article.find_all('div', class_='articulo-cuerpo')
        x = soup_article.find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)

    # df_features
    df_features = pd.DataFrame(
         {'Content': news_contents 
        })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
         'Article Link': list_links,
         'Newspaper': 'The Mirror'})
    
    return (df_features, df_show_info)

punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

def create_features_from_df(df):
    
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
        
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    
    wordnet_lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row]['Content_Parsed_4']
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    
    df['Content_Parsed_5'] = lemmatized_text_list
    
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
        
    df = df['Content_Parsed_6']
    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features

def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category

def predict_from_features(features):
        
    # Obtain the highest probability of the predictions for each article
    predictions_proba = svc_model.predict_proba(features).max(axis=1)    
    
    # Predict using the input model
    predictions_pre = svc_model.predict(features)

    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob, cat in zip(predictions_proba, predictions_pre):
        if prob > .65:
            predictions.append(cat)
        else:
            predictions.append(5)

    # Return result
    categories = [get_category_name(x) for x in predictions]
    
    return categories

def complete_df(df, categories):
    df['Prediction'] = categories
    return df

# Dash App

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

#####
# Edit from here
#####

# Colors
colors = {
    'background': '#ffffff',
    'text': '#696969',
    'header_table': '#ffedb3'
}

# Markdown text
markdown_text1 = '''

This application gathers the latest news from the newspapers **El Pais**, **The Guardian** and **The Mirror**, predicts their category between **Politics**, **Business**, **Entertainment**, **Sport**, **Tech** and **Other** and then shows a graphic summary.

The news categories are predicted with a Support Vector Machine model.

Please enter which newspapers would you like to scrape news off and press **Submit**:

'''

markdown_text2 = '''

*The scraped news are converted into a numeric feature vector with TF-IDF vectorization. Then, a Support Vector Classifier is applied to predict each category.*

Warning: The Mirror takes approximately 30 seconds to gather the news articles.

Created by Miguel Fernandez Zafra.
'''



app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    
    # Title
    html.H1(children='News Classification App',
            style={
                'textAlign': 'left',
                'color': colors['text'],
                'padding': '20px',
                'backgroundColor': colors['header_table']

            },
            className='banner',
           ),

    # Sub-title Left
    html.Div([
        dcc.Markdown(children=markdown_text1)],
        style={'width': '49%', 'display': 'inline-block'}),
    
    # Sub-title Right
    html.Div([
        dcc.Markdown(children=markdown_text2)],
        style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

    # Space between text and dropdown
    html.H1(id='space', children=' '),

    # Dropdown
    html.Div([
        dcc.Dropdown(
            options=[
                {'label': 'El Pais English', 'value': 'EPE'},
                {'label': 'The Guardian', 'value': 'THG'},
                {'label': 'The Mirror', 'value': 'TMI'}
            ],
            value=['EPE', 'THG'],
            multi=True,
            id='checklist')],
        style={'width': '40%', 'display': 'inline-block', 'float': 'left'}),
        

    # Button
    html.Div([
        html.Button('Submit', id='submit', type='submit')],
        style={'float': 'center'}),
    
    # Output Block
    html.Div(id='output-container-button', children=' '),
    
    # Graph1
    html.Div([
        dcc.Graph(id='graph1')],
        style={'width': '49%', 'display': 'inline-block'}),
    
    # Graph2
    html.Div([
        dcc.Graph(id='graph2')],
        style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
    
    # Table title
    html.Div(id='table-title', children='You can see a summary of the news articles below:'),

    # Space
    html.H1(id='space2', children=' '),
    
    # Table
    html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in ['Article Title', 'Article Link', 'Newspaper', 'Prediction']],
            style_data={'whiteSpace': 'normal'},
            style_as_list_view=True,
            style_cell={'padding': '5px', 'textAlign': 'left', 'backgroundColor': colors['background']},
            style_header={
                'backgroundColor': colors ['header_table'],
                'fontWeight': 'bold'
            },
            style_table={
                'maxHeight': '300',
                'overflowY':'scroll'
                
            },
            css=[{
                'selector': '.dash-cell div.dash-cell-value',
                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
            }]
        )],
        style={'width': '75%','float': 'left', 'position': 'relative', 'left': '12.5%', 'display': 'inline-block'}),    
        
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
    

])

@app.callback(
    Output('intermediate-value', 'children'),
    [],
    [State('checklist', 'value')],
    [Event('submit', 'click')])
def scrape_and_predict(values):
    
    df_features = pd.DataFrame()
    df_show_info = pd.DataFrame()
    
    if 'EPE' in values:
        # Get the scraped dataframes
        df_features = df_features.append(get_news_elpais()[0])
        df_show_info = df_show_info.append(get_news_elpais()[1])
    
    if 'THG' in values:
        df_features = df_features.append(get_news_theguardian()[0])
        df_show_info = df_show_info.append(get_news_theguardian()[1])
        
    if 'TMI' in values:
        df_features = df_features.append(get_news_themirror()[0])
        df_show_info = df_show_info.append(get_news_themirror()[1])

    df_features = df_features.reset_index().drop('index', axis=1)
    
    # Create features
    features = create_features_from_df(df_features)
    # Predict
    predictions = predict_from_features(features)
    # Put into dataset
    df = complete_df(df_show_info, predictions)
    # df.to_csv('Tableau Teaser/df_tableau.csv', sep='^')  # export to csv to work out an example in Tableau
    
    return df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('graph1', 'figure'),
    [Input('intermediate-value', 'children')])
def update_barchart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    # Create a summary df
    df_sum = df.groupby(['Newspaper', 'Prediction']).count()['Article Title']

    # Create x and y arrays for the bar plot for every newspaper
    if 'El Pais English' in df_sum.index:
    
        df_sum_epe = df_sum['El Pais English']
        x_epe = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_epe = [[df_sum_epe['politics'] if 'politics' in df_sum_epe.index else 0][0],
                [df_sum_epe['business'] if 'business' in df_sum_epe.index else 0][0],
                [df_sum_epe['entertainment'] if 'entertainment' in df_sum_epe.index else 0][0],
                [df_sum_epe['sport'] if 'sport' in df_sum_epe.index else 0][0],
                [df_sum_epe['tech'] if 'tech' in df_sum_epe.index else 0][0],
                [df_sum_epe['other'] if 'other' in df_sum_epe.index else 0][0]]   
    else:
        x_epe = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_epe = [0,0,0,0,0,0]
    
    if 'The Guardian' in df_sum.index:
        
        df_sum_thg = df_sum['The Guardian']
        x_thg = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_thg = [[df_sum_thg['politics'] if 'politics' in df_sum_thg.index else 0][0],
                [df_sum_thg['business'] if 'business' in df_sum_thg.index else 0][0],
                [df_sum_thg['entertainment'] if 'entertainment' in df_sum_thg.index else 0][0],
                [df_sum_thg['sport'] if 'sport' in df_sum_thg.index else 0][0],
                [df_sum_thg['tech'] if 'tech' in df_sum_thg.index else 0][0],
                [df_sum_thg['other'] if 'other' in df_sum_thg.index else 0][0]]   
    else:
        x_thg = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_thg = [0,0,0,0,0,0]

    if 'The Mirror' in df_sum.index:
    
        df_sum_tmi = df_sum['The Mirror']
        x_tmi = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_tmi = [[df_sum_tmi['politics'] if 'politics' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['business'] if 'business' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['entertainment'] if 'entertainment' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['sport'] if 'sport' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['tech'] if 'tech' in df_sum_tmi.index else 0][0],
                [df_sum_tmi['other'] if 'other' in df_sum_tmi.index else 0][0]]   

    else:
        x_tmi = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
        y_tmi = [0,0,0,0,0,0]

    # Create plotly figure
    figure = {
        'data': [
            {'x': x_epe, 'y':y_epe, 'type': 'bar', 'name': 'El Pais'},
            {'x': x_thg, 'y':y_thg, 'type': 'bar', 'name': 'The Guardian'},
            {'x': x_tmi, 'y':y_tmi, 'type': 'bar', 'name': 'The Mirror'}
        ],
        'layout': {
            'title': 'Number of news articles by newspaper',
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                    'color': colors['text']
            }
        }   
    }

    return figure

@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children')])
def update_piechart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    # Create a summary df
    df_sum = df['Prediction'].value_counts()

    # Create x and y arrays for the bar plot
    x = ['Politics', 'Business', 'Entertainment', 'Sport', 'Tech', 'Other']
    y = [[df_sum['politics'] if 'politics' in df_sum.index else 0][0],
         [df_sum['business'] if 'business' in df_sum.index else 0][0],
         [df_sum['entertainment'] if 'entertainment' in df_sum.index else 0][0],
         [df_sum['sport'] if 'sport' in df_sum.index else 0][0],
         [df_sum['tech'] if 'tech' in df_sum.index else 0][0],
         [df_sum['other'] if 'other' in df_sum.index else 0][0]]
    
    # Create plotly figure
    figure = {
        'data': [
            {'values': y,
             'labels': x, 
             'type': 'pie',
             'hole': .4,
             'name': '% of news articles'}
        ],
        
        'layout': {
            'title': '% of news articles',
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                    'color': colors['text']
            }
        }
        
    }
    
    return figure
    
    
@app.callback(
    Output('table', 'data'),
    [Input('intermediate-value', 'children')])
def update_table(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    data = df.to_dict('rows')
    return data

    
    
# Loading CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})



#####
# To here
#####

if __name__ == '__main__':
    app.run_server(debug=False)
