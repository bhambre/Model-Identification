from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask_cors import CORS, cross_origin


app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#clf = MultinomialNB(.04).fit(words,target)
pkl_file_type = open('strategyiden.pkl', 'rb')
clf = pickle.load(pkl_file_type)
#sample input https://strategybuilderapi.herokuapp.com/category?cash=0&convertable=0&depositoryreceipt=0&derivative=0&nonusbond=0&nonusstock=0&other=.0&preferredstock=0&usbond=0&usstock=0

@app.route('/category', methods=['get'])
def create_cm():
    Basic_Materials = request.args.get('Basic_Materials', 0)
    Cash_And_Equivalents = request.args.get('Cash_And_Equivalents', 0)
    Commercial_Services = request.args.get('Commercial_Services', 0)
    Communication_Services = request.args.get('Communication_Services', 0)
    Communications = request.args.get('Communications', 0)
    Consumer_Cyclical = request.args.get('Consumer_Cyclical', 0)
    Consumer_Defensive = request.args.get('Consumer_Defensive', 0)
    Consumer_Durables = request.args.get('Consumer_Durables', 0)
    Consumer_Non_Durables = request.args.get('Consumer_Non_Durables', 0)
    Consumer_Services = request.args.get('Consumer_Services', 0)
    Corporate_bonds = request.args.get('Corporate_bonds', 0)
    Derivative = request.args.get('Derivative', 0)
    Distribution_Services = request.args.get('Distribution_Services', 0)
    Electronic_Technology = request.args.get('Electronic_Technology', 0)
    Energy = request.args.get('Energy', 0)
    Energy_Minerals = request.args.get('Energy_Minerals', 0)
    Finance = request.args.get('Finance', 0)
    Financial_Services = request.args.get('Financial_Services', 0)
    Government = request.args.get('Government', 0)
    Government_bonds = request.args.get('Government_bonds', 0)
    Health_Services = request.args.get('Health_Services', 0)
    Health_Technology = request.args.get('Health_Technology', 0)
    Healthcare = request.args.get('Healthcare', 0)
    Industrial_Services = request.args.get('Industrial_Services', 0)
    Industrials = request.args.get('Industrials', 0)
    Miscellaneous = request.args.get('Miscellaneous', 0)
    Municipal_bonds = request.args.get('Municipal_bonds', 0)
    Non_Energy_Minerals = request.args.get('Non_Energy_Minerals', 0)
    Process_Industries = request.args.get('Process_Industries', 0)
    Producer_Manufacturing = request.args.get('Producer_Manufacturing', 0)
    Real_Estate = request.args.get('Real_Estate', 0)
    Retail_Trade = request.args.get('Retail_Trade', 0)
    Securitized_products = request.args.get('Securitized_products', 0)
    Technology = request.args.get('Technology', 0)
    Technology_Services = request.args.get('Technology_Services', 0)
    Transportation = request.args.get('Transportation', 0)
    Unknown = request.args.get('Unknown', 0)
    Utilities = request.args.get('Utilities', 0)
    # use default value repalce 'None'
    #change = request.args.get('change', None)
    feat = [[Basic_Materials, Cash_And_Equivalents, Commercial_Services, Communication_Services, Communications, Consumer_Cyclical, Consumer_Defensive, Consumer_Durables, Consumer_Non_Durables, Consumer_Services, Corporate_bonds, Derivative, Distribution_Services, Electronic_Technology, Energy, Energy_Minerals, Finance, Financial_Services, Government, Government_bonds, Health_Services, Health_Technology, Healthcare, Industrial_Services, Industrials, Miscellaneous, Municipal_bonds, Non_Energy_Minerals, Process_Industries, Producer_Manufacturing, Real_Estate, Retail_Trade, Securitized_products, Technology, Technology_Services, Transportation, Unknown, Utilities]] 
    
    catprob = int((clf.predict_proba(feat).max(axis=1)[0]) * 100)
    cat = str(clf.predict(feat)[0])

    if catprob < 15:
    	cat = 'Unable to identify' 

    result = {'category': cat, 'category score':catprob} 


        
    return jsonify(result)

if __name__ == '__main__':
     app.run()

     #test123