import pandas as pd
import numpy as np
import sqlite3

# show all available columns
pd.set_option('display.max_columns', 500)
# show all available rows
pd.set_option('display.max_rows', 500)

# models
import statsmodels.formula.api as smf
import statsmodels.api as sm

# regression metrics
from sklearn.metrics import accuracy_score

# warnig treatments
import warnings
warnings.filterwarnings('ignore')

# ensemble models 
from xgboost import XGBRegressor, XGBClassifier

# statistics
from scipy.stats import loguniform, uniform

# import model
import pickle

# system
import sys
import argparse 
import os

from flask import Flask, request, render_template

# instancia o objeto do Flask
app = Flask(__name__)

def df_return(df_res_predict, pred_prob, pred_res):
    
    df = df_res_predict[['Game_Date', 'HomeTeam', 'AwayTeam']]
    
    df['Prob_Win_HomeTeam'] = pred_prob[:, 0]
    df['Prob_Draw'] = pred_prob[:, 1]
    df['Prob_Win_AwayTeam'] = pred_prob[:, 2]
    
    df['Result'] = pred_res
    
    df['Result'].loc[ df['Result'] == 0 ] = 'Home team win'
    df['Result'].loc[ df['Result'] == 1 ] = 'Draw'
    df['Result'].loc[ df['Result'] == 2 ] = 'Away team win'
    
    return df    


def predict_result(GameDate01, GameDate02):
    
    # conexão com o BD Premier League - AWS
    dbname = 'premierleague.cqoq1gvjbsxj.us-east-2.rds.amazonaws.com'
    db_PremierLeague = sqlite3.connect(dbname)

    query = """ SELECT * FROM PremierLeague_Info"""
    
    df_result = pd.read_sql(query, db_PremierLeague)
    
    df_res_predict = df_result.loc[(df_result['Game_Date']>=GameDate01) & (df_result['Game_Date']<=GameDate02)]
        
    # Carregar modelo
    with open('premier_xgboost.pkl', 'rb') as f:
        xb = pickle.load(f)
    
    pred_prob = xb.predict_proba(df_res_predict)*100
    
    pred_res = xb.predict(df_res_predict)
    
    df_res_predict_final = df_return(df_res_predict, pred_prob, pred_res)
    
    return pred_prob, pred_res, df_res_predict_final

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    GameDate01 = request.form['GameDate01']
    GameDate02 = request.form['GameDate02']
    
    pred_prob, pred_res, df_res_predict = predict_result(GameDate01, GameDate02)
    response = df_res_predict

    return response.to_html()


if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 5001))
    # app.run(host='0.0.0.0', port=port)
    app.run(port=5000,host='0.0.0.0')