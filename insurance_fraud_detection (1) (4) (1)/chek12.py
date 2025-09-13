import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
import pickle
#load & view raw data
data1=['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year', 'policy_state_IL', 'policy_state_IN', 'policy_state_OH', 'policy_csl_100/300', 'policy_csl_250/500', 'policy_csl_500/1000', 'insured_sex_FEMALE', 'insured_sex_MALE', 'insured_education_level_Associate', 'insured_education_level_College', 'insured_education_level_High School', 'insured_education_level_JD', 'insured_education_level_MD', 'insured_education_level_Masters', 'insured_education_level_PhD', 'insured_occupation_adm-clerical', 'insured_occupation_armed-forces', 'insured_occupation_craft-repair', 'insured_occupation_exec-managerial', 'insured_occupation_farming-fishing', 'insured_occupation_handlers-cleaners', 'insured_occupation_machine-op-inspct', 'insured_occupation_other-service', 'insured_occupation_priv-house-serv', 'insured_occupation_prof-specialty', 'insured_occupation_protective-serv', 'insured_occupation_sales', 'insured_occupation_tech-support', 'insured_occupation_transport-moving', 'insured_hobbies_base-jumping', 'insured_hobbies_basketball', 'insured_hobbies_board-games', 'insured_hobbies_bungie-jumping', 'insured_hobbies_camping', 'insured_hobbies_chess', 'insured_hobbies_cross-fit', 'insured_hobbies_dancing', 'insured_hobbies_exercise', 'insured_hobbies_golf', 'insured_hobbies_hiking', 'insured_hobbies_kayaking', 'insured_hobbies_movies', 'insured_hobbies_paintball', 'insured_hobbies_polo', 'insured_hobbies_reading', 'insured_hobbies_skydiving', 'insured_hobbies_sleeping', 'insured_hobbies_video-games', 'insured_hobbies_yachting', 'insured_relationship_husband', 'insured_relationship_not-in-family', 'insured_relationship_other-relative', 'insured_relationship_own-child', 'insured_relationship_unmarried', 'insured_relationship_wife', 'incident_type_Multi-vehicle Collision', 'incident_type_Parked Car', 'incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft', 'collision_type_?', 'collision_type_Front Collision', 'collision_type_Rear Collision', 'collision_type_Side Collision', 'incident_severity_Major Damage', 'incident_severity_Minor Damage', 'incident_severity_Total Loss', 'incident_severity_Trivial Damage', 'authorities_contacted_Ambulance', 'authorities_contacted_Fire', 'authorities_contacted_None', 'authorities_contacted_Other', 'authorities_contacted_Police', 'incident_state_NC', 'incident_state_NY', 'incident_state_OH', 'incident_state_PA', 'incident_state_SC', 'incident_state_VA', 'incident_state_WV', 'incident_city_Arlington', 'incident_city_Columbus', 'incident_city_Hillsdale', 'incident_city_Northbend', 'incident_city_Northbrook', 'incident_city_Riverwood', 'incident_city_Springfield', 'property_damage_?', 'property_damage_NO', 'property_damage_YES', 'police_report_available_?', 'police_report_available_NO', 'police_report_available_YES', 'auto_make_Accura', 'auto_make_Audi', 'auto_make_BMW', 'auto_make_Chevrolet', 'auto_make_Dodge', 'auto_make_Ford', 'auto_make_Honda', 'auto_make_Jeep', 'auto_make_Mercedes', 'auto_make_Nissan', 'auto_make_Saab', 'auto_make_Suburu', 'auto_make_Toyota', 'auto_make_Volkswagen', 'auto_model_3 Series', 'auto_model_92x', 'auto_model_93', 'auto_model_95', 'auto_model_A3', 'auto_model_A5', 'auto_model_Accord', 'auto_model_C300', 'auto_model_CRV', 'auto_model_Camry', 'auto_model_Civic', 'auto_model_Corolla', 'auto_model_E400', 'auto_model_Escape', 'auto_model_F150', 'auto_model_Forrestor', 'auto_model_Fusion', 'auto_model_Grand Cherokee', 'auto_model_Highlander', 'auto_model_Impreza', 'auto_model_Jetta', 'auto_model_Legacy', 'auto_model_M5', 'auto_model_MDX', 'auto_model_ML350', 'auto_model_Malibu', 'auto_model_Maxima', 'auto_model_Neon', 'auto_model_Passat', 'auto_model_Pathfinder', 'auto_model_RAM', 'auto_model_RSX', 'auto_model_Silverado', 'auto_model_TL', 'auto_model_Tahoe', 'auto_model_Ultima', 'auto_model_Wrangler', 'auto_model_X5', 'auto_model_X6']
print(len(data1))
data11 = pd.DataFrame(0,index=np.arange(1),columns=data1)
for i in data11:
    print(i)
data = pd.read_csv('./testing.csv')
fdata=pd.DataFrame()
#print(data.dtypes)
original_data = data.copy()
#Remove Less Correlated Columns
deleteCols = ["policy_number", "policy_bind_date", "insured_zip", "incident_location", "incident_date"]
data = data.drop(deleteCols, axis=1)
list_hot_encoded = []
for column in data.columns:
    if(data[column].dtypes==object):
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        list_hot_encoded.append(column)
#Drop hot-encoded columns
data = data.drop(list_hot_encoded, axis=1)
print(len(data.columns))


df3 = data11.assign(**data.iloc[0])
df3 = df3.drop('insured_sex_Male', axis=1)
df3 = df3.drop('property_damage_Yes', axis=1)
df3 = df3.drop('police_report_available_Yes', axis=1)
for i in df3:
    print(i)
print(len(df3))
model = pickle.load(open('model1.pkl', 'rb'))
array_features = [np.array(df3.iloc[0])]
prediction = model.predict(array_features)
print(prediction)


##for column in data11:
##    if (data11[column]) ==(data[column]):
##        fdata[column]=data[column]
##        print(fdata[column])
##    else:
##        fdata[column]=0


##for i in range(0,len(data1)):
##    if i< len(data.columns):
##        print(data.columns[i])
##        if data.columns[i]== data1[i]:
##            pass
##        else:
##            data[i]=0
##            print(data.i)
##    else:
##        data.i=0
##        print(data.i)

