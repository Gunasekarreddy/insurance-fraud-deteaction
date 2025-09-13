import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics

# Define model features (same order as during training)
data1=['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year', 'policy_state_IL', 'policy_state_IN', 'policy_state_OH', 'policy_csl_100/300', 'policy_csl_250/500', 'policy_csl_500/1000', 'insured_sex_FEMALE', 'insured_sex_MALE', 'insured_education_level_Associate', 'insured_education_level_College', 'insured_education_level_High School', 'insured_education_level_JD', 'insured_education_level_MD', 'insured_education_level_Masters', 'insured_education_level_PhD', 'insured_occupation_adm-clerical', 'insured_occupation_armed-forces', 'insured_occupation_craft-repair', 'insured_occupation_exec-managerial', 'insured_occupation_farming-fishing', 'insured_occupation_handlers-cleaners', 'insured_occupation_machine-op-inspct', 'insured_occupation_other-service', 'insured_occupation_priv-house-serv', 'insured_occupation_prof-specialty', 'insured_occupation_protective-serv', 'insured_occupation_sales', 'insured_occupation_tech-support', 'insured_occupation_transport-moving', 'insured_hobbies_base-jumping', 'insured_hobbies_basketball', 'insured_hobbies_board-games', 'insured_hobbies_bungie-jumping', 'insured_hobbies_camping', 'insured_hobbies_chess', 'insured_hobbies_cross-fit', 'insured_hobbies_dancing', 'insured_hobbies_exercise', 'insured_hobbies_golf', 'insured_hobbies_hiking', 'insured_hobbies_kayaking', 'insured_hobbies_movies', 'insured_hobbies_paintball', 'insured_hobbies_polo', 'insured_hobbies_reading', 'insured_hobbies_skydiving', 'insured_hobbies_sleeping', 'insured_hobbies_video-games', 'insured_hobbies_yachting', 'insured_relationship_husband', 'insured_relationship_not-in-family', 'insured_relationship_other-relative', 'insured_relationship_own-child', 'insured_relationship_unmarried', 'insured_relationship_wife', 'incident_type_Multi-vehicle Collision', 'incident_type_Parked Car', 'incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft', 'collision_type_?', 'collision_type_Front Collision', 'collision_type_Rear Collision', 'collision_type_Side Collision', 'incident_severity_Major Damage', 'incident_severity_Minor Damage', 'incident_severity_Total Loss', 'incident_severity_Trivial Damage', 'authorities_contacted_Ambulance', 'authorities_contacted_Fire', 'authorities_contacted_None', 'authorities_contacted_Other', 'authorities_contacted_Police', 'incident_state_NC', 'incident_state_NY', 'incident_state_OH', 'incident_state_PA', 'incident_state_SC', 'incident_state_VA', 'incident_state_WV', 'incident_city_Arlington', 'incident_city_Columbus', 'incident_city_Hillsdale', 'incident_city_Northbend', 'incident_city_Northbrook', 'incident_city_Riverwood', 'incident_city_Springfield', 'property_damage_?', 'property_damage_NO', 'property_damage_YES', 'police_report_available_?', 'police_report_available_NO', 'police_report_available_YES', 'auto_make_Accura', 'auto_make_Audi', 'auto_make_BMW', 'auto_make_Chevrolet', 'auto_make_Dodge', 'auto_make_Ford', 'auto_make_Honda', 'auto_make_Jeep', 'auto_make_Mercedes', 'auto_make_Nissan', 'auto_make_Saab', 'auto_make_Suburu', 'auto_make_Toyota', 'auto_make_Volkswagen', 'auto_model_3 Series', 'auto_model_92x', 'auto_model_93', 'auto_model_95', 'auto_model_A3', 'auto_model_A5', 'auto_model_Accord', 'auto_model_C300', 'auto_model_CRV', 'auto_model_Camry', 'auto_model_Civic', 'auto_model_Corolla', 'auto_model_E400', 'auto_model_Escape', 'auto_model_F150', 'auto_model_Forrestor', 'auto_model_Fusion', 'auto_model_Grand Cherokee', 'auto_model_Highlander', 'auto_model_Impreza', 'auto_model_Jetta', 'auto_model_Legacy', 'auto_model_M5', 'auto_model_MDX', 'auto_model_ML350', 'auto_model_Malibu', 'auto_model_Maxima', 'auto_model_Neon', 'auto_model_Passat', 'auto_model_Pathfinder', 'auto_model_RAM', 'auto_model_RSX', 'auto_model_Silverado', 'auto_model_TL', 'auto_model_Tahoe', 'auto_model_Ultima', 'auto_model_Wrangler', 'auto_model_X5', 'auto_model_X6']


# Create a zero-filled DataFrame with model's feature set
data11 = pd.DataFrame(0, index=[0], columns=data1)

# Load test data
df = pd.read_csv('./testing.csv')
df[['insured_zip']] = df[['insured_zip']].astype(object)

# Create vehicle age feature
df['vehicle_age'] = 2018 - df['auto_year']

# Time binning
bins = [-1, 3, 6, 9, 12, 17, 20, 24]
names = ["past_midnight", "early_morning", "morning", 'fore-noon', 'afternoon', 'evening', 'night']
df['incident_period_of_day'] = pd.cut(df.incident_hour_of_the_day, bins, labels=names).astype(object)

# Drop unneeded columns
df.drop(columns=[
    'policy_number', 'insured_zip', 'policy_bind_date', 'incident_date',
    'incident_location', 'auto_year', 'incident_hour_of_the_day'
], inplace=True)

# Check and handle unknowns ('?')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].replace('?', np.nan)

# Create dummy variables for categorical features
dummies = pd.get_dummies(df[[
    'policy_state', 'policy_csl', 'insured_sex', 'insured_education_level',
    'insured_occupation', 'insured_hobbies', 'insured_relationship',
    'incident_type', 'incident_severity', 'authorities_contacted',
    'incident_state', 'incident_city', 'auto_make', 'auto_model',
    'incident_period_of_day'
]])

# Join binary features
dummies = dummies.join(df[['collision_type', 'property_damage', 'police_report_available']])

# Encode and clean target categorical columns
dummies['collision_type'] = LabelEncoder().fit_transform(dummies['collision_type'].astype(str))
dummies['property_damage'] = dummies['property_damage'].map({'Yes': 1, 'No': 0, np.nan: 0})
dummies['police_report_available'] = dummies['police_report_available'].map({'Yes': 1, 'No': 0, np.nan: 0})

# Combine with numeric features
numeric_data = df.select_dtypes(include=[np.number])
X = pd.concat([dummies.drop(columns=[]), numeric_data], axis=1)

# Remove duplicate columns if any
X = X.loc[:, ~X.columns.duplicated()]

# Ensure only model features are selected
available_features = X.columns.intersection(data1)
missing_features = set(data1) - set(available_features)

# Fill matching features
df3 = data11.copy()
df3.loc[0, available_features] = X.loc[0, available_features]

# If needed, warn about any missing columns
if missing_features:
    print("Warning: The following features are missing from the test input and filled with 0:")
    print(missing_features)

# Load trained model


# Predict using prepared input
input_array = [df3.iloc[0].values]
prediction = model.predict(input_array)

# Output result
print("Prediction:", prediction)
