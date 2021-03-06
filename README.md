# Loan Default Prediction
The objective in this project is to determing whether a person will default in repaying a certain loan amount based on their financial and some personal details. The dataset is obtained from Kaggle https://www.kaggle.com/c/credit-default-prediction-ai-big-data which is an ogoing competition. 

## **Personal Overview**:  
As per my understanding, one of the key challenge in this competition lies in the imbalanced nature of the datasets which is probably common to
classification problems. I found this to be true here particularly after facing continuous struggle in improving the model accuracy prior to applying SMOTE. It is worth mentioning that the evalation metric used to rank a submission is the f1 score which significantly varies upto fourth term after decimal. The highest score at present is about 0.5643 which involves a highly sophisticated model. My experiments have primarily been with CatBoost, XGBoost and RandomForest classifier and after a significant number of trials, I achieved a score 0f 0.48571 with RandomForest which ranks in 82nd position. A rather peculiar observation I had is that majority of the times, the actual score of the model measured by Kaggle did not correlate with its performance on the test dataset prepared from train-test split. 

## Workflow 
### Data preprocessing : 
* Numerical features with relatively small number of missing values are replaced by the median value since most of them have a skewed distribution, while columns having more than 50 percent missing values are dropped. Categorical features are treated with the value having highest frequency.
* Two additional features are engineered from the existing feature list having higher correlation with the target variable. Also, categorical features with multiple categories are redefined through some general understanding of the domain.
* Outliers are analysed carefully and handled though IQR technique and examined through boxplots
* Implemented XGBoost, CatBoost and RandomForest Classfier and found that model accuracy improved after applying SMOTE in the dataset and best performance is shown by RandomForest

### Deployment:
* Front-end prepared using generic HTML template
* Back-end developed using Flask and deployed in Heroku - https://loandefpredic.herokuapp.com/


