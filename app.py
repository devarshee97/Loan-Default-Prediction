import os

from flask import Flask, request, render_template
from flask_cors import cross_origin
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('random_forest_classifier.pkl', 'rb'))


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":


        Annual_Income = int(request.form['Annual_Income'])
        Years_in_current_job = int(request.form['Years_in_current_job'])
        Tax_Liens = int(request.form['Tax_Liens'])
        Open_accounts = int(request.form["Open_accounts"])
        Years_of_credit_history = int(request.form['Years_of_credit_history'])
        Maximum_open_credit  = int(request.form['Maximum_open_credit'])
        Number_of_Credit_Problems = int(request.form['Number_of_credit_problems'])
        Bankruptcies = int(request.form['Bankruptcies'])
        Current_Loan_Amount = int(request.form['Current_Loan_Amount'])
        Current_Credit_Balance = int(request.form['Current_Credit_Balance'])
        Monthly_Debt = int(request.form['Monthly_Debt'])
        Credit_Score = int(request.form['Credit_Score'])
        Monthly_amount_left = Annual_Income - Monthly_Debt
        Residual_debt = Current_Loan_Amount - Current_Credit_Balance
        Purpose_label = request.form['Purpose_label']

        if(Purpose_label == 'Recreation'):
            Recreation = 1
            Loan_Debt = 0
            Personal = 0
            other = 0
        elif (Purpose_label == "Loan_Debt"):
            Recreation = 0
            Loan_Debt = 1
            Personal = 0
            other = 0
        elif (Purpose_label == "Personal"):
            Recreation = 0
            Loan_Debt = 0
            Personal = 1
            other = 0
        elif (Purpose_label == "other"):
            Recreation = 0
            Loan_Debt = 0
            Personal = 0
            other = 1
        else:
            Recreation = 0
            Loan_Debt = 0
            Personal = 0
            other = 0

        Loan_Duration = request.form["Loan_Duration"]

        if (Loan_Duration == "Short_Term"):
            Short_Term = 1
        else:
            Short_Term = 0

        Home_ownership = request.form["Home_Ownership"]
        if(Home_ownership == "Own_Home"):
            Own_Home =  1
            Home_Mortgage = 0
            Rent = 0
        elif (Home_ownership == "Home_Mortgage"):
            Own_Home  = 0
            Home_Mortgage = 1
            Rent = 0

        elif (Home_ownership == "Rent"):
            Own_Home = 0
            Home_Mortgage = 0
            Rent = 1
        else:
            Own_Home = 0
            Home_Mortgage = 0
            Rent = 0



        Loan_status = model.predict([[Years_in_current_job, Tax_Liens, Open_accounts,
        Years_of_credit_history, Maximum_open_credit,
        Number_of_Credit_Problems, Bankruptcies, Credit_Score,
        Monthly_amount_left, Residual_debt, Home_Mortgage, Own_Home, Rent,
        Short_Term,
        Loan_Debt, Personal,
        Recreation, other]])

        if int(Loan_status) == 1:
            Status = "Loan may be defaulter"
        else:
            Status = "Loan will be successfully repayed"

        return render_template("index.html", prediction_text = Status)

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



