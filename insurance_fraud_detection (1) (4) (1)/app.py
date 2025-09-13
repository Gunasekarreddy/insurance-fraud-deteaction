from flask import Flask, redirect, request, render_template, flash, url_for
import os

import csv
import pymongo
import pandas as pd
import numpy as np
import pickle

s = 0

# client = pymongo.MongoClient("mongodb+srv://Sivakumar23:siva2310@atlascluster.tig0e4k.mongodb.net/?retryWrites=true&w=majority")
client = pymongo.MongoClient("mongodb+srv://gunasekar:gunasekar@shopez.x9dqx.mongodb.net/?retryWrites=true&w=majority&appName=shopEZ")
# db = client.Farud_detection
db = client["Fraud_detection"]
c1 = db["users"]
c2 = db["files"]


flag1 = 0
database = []

app = Flask(__name__)

def newpredict():
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
##    df3 = df3.drop('insured_sex_Male', axis=1)
##    df3 = df3.drop('property_damage_Yes', axis=1)
##    df3 = df3.drop('police_report_available_Yes', axis=1)
    for i in df3:
        print(i)
    print(len(df3))
    model = pickle.load(open('model1.pkl', 'rb'))
    array_features = [np.array(df3.iloc[0])]
    prediction = model.predict(array_features)
    print(type(prediction[0]))
    return prediction[0]
   


    

@app.route('/')
def home_page(title='Welcome to the Login Form Demo!'):
    print('1')

    return render_template('landing.html', title=title, ok='run')


@app.route('/signin', methods=['GET', 'POST'])
def login_page(title='Login Demo'):
    global msgs, error
    error = []
    if request.method == 'GET':
        return render_template('landing.html', ok='run')

    elif request.method == 'POST':
        print('3')
        print(request.form['email'], request.form['password'])
        item_details = c1.find({"email": request.form['email']})
        for item in item_details:
            if request.form['email'] == item['email'] and request.form['password'] == item['password']:
                msgs = 'Login Successfuly'
                print("flash")
                return render_template('landing.html', msgs=msgs, ok='ok')
        error = 'Email or Password Incorrect'
        print(error)
    return render_template('landing.html', error=error, ok='run')


@app.route('/signup', methods=['GET', 'POST'])
def signup_page(title='Signup Page'):
    global flag1, war, msgs
    error = ''
    war = ''
    msgs = ''
    user_data = {}

    if request.method == 'GET':
        flag1 = 0
        return render_template('landing.html', error=error, title=title, ok='run')

    elif request.method == 'POST':
        if request.form['password'] != request.form['confirm_password']:
            error = 'Password do not match'
            return render_template('landing.html', error=error, ok='run')

        if request.form['email'] != '' and request.form['fname'] != '' and request.form['lname'] != '' and request.form['password'] != '' and request.form['confirm_password'] != '':
            user_data = dict(request.form)
            print(user_data)
            item_details = c1.find({"email": request.form['email']})
            for item in item_details:
                if request.form['email'] == item['email']:
                    war = 'Email Already Registered'
                    return render_template('landing.html', war=war, ok='run')
            if flag1 == 0:
                item_1 = {
                    "Username": user_data['fname']+user_data['lname'],
                    "email": user_data['email'],
                    "password": user_data['password'],
                    "cpassword": user_data['confirm_password']
                }
                c1.insert_one(item_1)
                msgs = 'Registration Successfully'
                return render_template('landing.html', msgs=msgs, ok='run')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global coll, newc, database
    error = ''
    war = ''
    msgs = ''
    if request.method == 'GET':
        return render_template('landing.html', ok='run')
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        df = pd.read_csv(f)  # csv file which you want to import
        records_ = df.to_dict(orient='records')
        database = db.list_collection_names()
        print(records_)
# for s in range (0,len(database)):
# if database[s] == filename:
# war="Please Change File Name"
# return render_template('landing.html',war=war)
# else:
        collection1 = db[f.filename]
        collection1.insert_many(records_)
        msgs = 'Upload Successfully'

        return render_template('landing.html', msgs=msgs, ok='ok1', name=f.filename)
    else:
        error = 'Invalid Request'
        return render_template('landing.html', error=error)


@app.route('/download', methods=['GET', 'POST'])
def download():
    results = []
    if request.method == 'GET':
        return render_template('landing.html', ok='run')
    if request.method == 'POST':
        filename = request.form['filename']
        # print(filename)
        strfile = str(filename)
        cursor = db[strfile].find({}, {'_id': False})
        for document in cursor:
            print(document)
            results.append(dict(document))
            fieldnames = [key for key in results[0].keys()]
        return render_template('home.html', msgs='Download Successfully', results=results, fieldnames=fieldnames, len=len)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    msgs = ''
    war = ''
    if request.method == 'GET':
        war = "No File Found"
        return render_template('landing.html', war=war, ok='run')

    if request.method == 'POST':

        msgs = "Predict Successfully"
        return render_template('landing.html', msgs=msgs, ok='ok')


@app.route('/claimcheck1', methods=['GET', 'POST'])
def claimcheck1():
    return render_template('reg_page.html', ok='run')
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    return render_template('landing.html', ok='run')

@app.route('/result', methods=['GET', 'POST'])
def images():
    return render_template('images.html', ok='run')
@app.route('/claimcheck', methods=['GET', 'POST'])
def claimcheck():
    if request.method == 'GET':
        war = "No File Found"
        return render_template('reg_page.html', war=war, ok='run')

    if request.method == 'POST':
        item = {
            "months_as_customer"        : int(request.form['1']),
            "age"                       : int(request.form['2']),
            "policy_number"             : int(request.form['3']),
            "policy_bind_date"          : request.form['4'],
            "policy_state"              : request.form['5'],
            "policy_csl"                : request.form['6'],
            "policy_deductable"         : int(request.form['7']),
            "policy_annual_premium"     : float(request.form['8']),
            "umbrella_limit"            : int(request.form['9']),
            "insured_zip"               : int(request.form['10']),
            "insured_sex"               : request.form['11'],
            "insured_education_level"   : request.form['12'],
            "insured_occupation"        : request.form['13'],
            "insured_hobbies"           : request.form['14'],
            "insured_relationship"      : request.form['15'],
            "capital-gains"             : int(request.form['16']),
            "capital-loss"              : int(request.form['17']),
            "incident_date"             : request.form['18'],
            "incident_type"             : request.form['19'],
            "collision_type"            : request.form['20'],
            "incident_severity"         : request.form['21'],
            "authorities_contacted"     : request.form['22'],
            "incident_state"            : request.form['23'],
            "incident_city"             : request.form['24'],
            "incident_location"         : request.form['25'],
            "incident_hour_of_the_day"  : int(request.form['26']),
            "number_of_vehicles_involved": int(request.form['27']),
            "property_damage"           : request.form['28'],
            "bodily_injuries"           : int(request.form['29']),
            "witnesses"                 : int(request.form['30']),
            'police_report_available'   : request.form['31'],
            "total_claim_amount"        : int(request.form['32']),
            "injury_claim"              : int(request.form['33']),
            "property_claim"            : int(request.form['34']),
            "vehicle_claim"             : int(request.form['35']),
            "auto_make"                 : request.form['36'],
            "auto_model"                : request.form['37'],
            "auto_year"                 : int(request.form['38'])
        }
        print(item)
        df = pd.DataFrame(item,index=[0])
        df.to_csv('./testing.csv',index=False)
        #print(df.dtypes)
        #print(df)
        xa=newpredict()
        if xa==0:
            print('1')
            pmsgs='https://freepngimg.com/save/27880-green-tick-clipart/1400x1600' #default msg
            Ack='Thank You !'
            msgs="ORGINAL CLAIM"
        else:
            print('2')
            pmsgs='https://www.kindpng.com/picc/m/413-4131735_transparent-tick-mark-png-transparent-background-transparent-background.png' #default msg
            Ack='Sorry !!!'
            msgs="FRUAD CLAIM"
    return render_template('popup.html',msgs=msgs,pmsgs=pmsgs,Ack=Ack,ok='run')
    

if (__name__ == '__main__'):
    app.secret_key = "abc123"
    app.run(host='0.0.0.0', port=7000)
