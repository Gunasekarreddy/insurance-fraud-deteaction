import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
import pickle
#df = pd.read_csv('./insuranceFraud_Dataset.csv')
df = pd.read_csv('./testing.csv')
df.nunique()
df[['insured_zip']] = df[['insured_zip']].astype(object)
df.auto_year.value_counts()  # check the spread of years to decide on further action.
df['vehicle_age'] = 2018 - df['auto_year'] # Deriving the age of the vehicle based on the year value 
df['vehicle_age'].head(10)
bins = [-1, 3, 6, 9, 12, 17, 20, 24]  # Factorize according to the time period of the day.
names = ["past_midnight", "early_morning", "morning", 'fore-noon', 'afternoon', 'evening', 'night']
df['incident_period_of_day'] = pd.cut(df.incident_hour_of_the_day, bins, labels=names).astype(object)
df[['incident_hour_of_the_day', 'incident_period_of_day']].head(20)
# Check on categorical variables:
df.select_dtypes(include=['object']).columns  # checking categorcial columns
df = df.drop(columns = [
    'policy_number', 
    'insured_zip', 
    'policy_bind_date', 
    'incident_date', 
    'incident_location', 
    'auto_year', 
    'incident_hour_of_the_day'])
unknowns = {}
for i in list(df.columns):
    if (df[i]).dtype == object:
        j = np.sum(df[i] == "?")
        unknowns[i] = j
unknowns = pd.DataFrame.from_dict(unknowns, orient = 'index')
df.collision_type.value_counts()

df.police_report_available.value_counts()
df.select_dtypes(include=['object']).columns  # checking categorcial columns

dummies = pd.get_dummies(df[[
    'policy_state', 
    'policy_csl', 
    'insured_sex', 
    'insured_education_level',
    'insured_occupation', 
    'insured_hobbies', 
    'insured_relationship',
    'incident_type', 
    'incident_severity',
    'authorities_contacted', 
    'incident_state', 
    'incident_city',
    'auto_make', 
    'auto_model',
    'incident_period_of_day']])
dummies = dummies.join(df[[
    'collision_type', 
    'property_damage', 
    'police_report_available'
    ]])
X = dummies.iloc[:, 0:]
y = dummies.iloc[:, -1]
print(X)
from sklearn.preprocessing import LabelEncoder
X['collision_en'] = LabelEncoder().fit_transform(dummies['collision_type'])
X[['collision_type', 'collision_en']]
X['property_damage'].replace(to_replace='YES', value=1, inplace=True)
X['property_damage'].replace(to_replace='NO', value=0, inplace=True)
X['property_damage'].replace(to_replace='?', value=0, inplace=True)
X['police_report_available'].replace(to_replace='YES', value=1, inplace=True)
X['police_report_available'].replace(to_replace='NO', value=0, inplace=True)
X['police_report_available'].replace(to_replace='?', value=0, inplace=True)
X = X.drop(columns = ['collision_type'])
X = pd.concat([X, df._get_numeric_data()], axis=1)  # joining numeric columns
print(X.iloc[0])


model = pickle.load(open('model1.pkl', 'rb'))
array_features = [np.array(X.iloc[0])]
prediction = model.predict(array_features)
print(prediction)
