from flask import Flask,redirect,request,render_template,flash,url_for
import os
from distutils.log import debug
import csv
import pymongo
import pandas as pd
import numpy as np

s=0

client = pymongo.MongoClient("mongodb+srv://gunasekar:gunasekar@shopez.x9dqx.mongodb.net/?retryWrites=true&w=majority&appName=shopEZ")
#db = client.Farud_detection
db=client["Fraud_detection"]
c1= db["users"]
c2 = db["files"]


flag1=0
database=[]

app = Flask(__name__)

def fun():
    global database
    database= db.list_collection_names()  
    #print(database)

fun()


@app.route('/')
def home_page(title='Welcome to the Login Form Demo!'):
    print('1')
    
    return render_template('landing.html', title=title,ok='run')


@app.route('/signin', methods=['GET', 'POST'])
def login_page(title='Login Demo'):
    global msgs,error
    error=[]
    if request.method == 'GET':
        return render_template('landing.html',ok='run')

    elif request.method == 'POST':
        print('3')
        print(request.form['email'],request.form['password'])
        item_details = c1.find({"email" : request.form['email']})
        for item in item_details:
            if request.form['email']== item ['email'] and request.form['password'] == item ['password']:
                msgs='Login Successfuly'
                print("flash")
                return render_template('landing.html',msgs=msgs,ok='ok')
        error='Email or Password Incorrect'
        print(error)
    return render_template('landing.html',error=error,ok='run')


@app.route('/signup', methods=['GET', 'POST'])
def signup_page(title='Signup Page'):
    global flag1,war,msgs
    error = ''
    war=''
    msgs=''
    user_data = {}

    if request.method == 'GET':
        flag1=0
        return render_template('landing.html', error=error, title=title,ok='run')

    elif request.method == 'POST':
        if request.form['password'] != request.form['confirm_password']:
            error='Password do not match'
            return render_template('landing.html', error=error,ok='run')

        if request.form['email'] != '' and request.form['fname'] != '' and request.form['lname'] != '' and request.form['password'] != '' and request.form['confirm_password'] != '':
            user_data = dict(request.form)
            print(user_data)
            item_details = c1.find({"email" : request.form['email']})
            for item in item_details:
                if request.form['email']== item['email']:
                    war='Email Already Registered'
                    return render_template('landing.html', war=war,ok='run')
            if flag1==0:
                item_1 = {
                    "Username" : user_data['fname']+user_data['lname'],
                          "email" : user_data['email'],
                          "password" : user_data['password'],
                          "cpassword" : user_data['confirm_password']
                              }
                c1.insert_one(item_1)
                msgs='Registration Successfully'
                return render_template('landing.html', msgs=msgs,ok='run')
            
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global coll, newc,database
    error = ''
    war=''
    msgs=''
    if request.method == 'GET':
        return render_template('landing.html',ok='run')
    if request.method == 'POST':
        f =request.files['file']
        filename=f.filename
        df = pd.read_csv(f) #csv file which you want to import 
        records_= df.to_dict(orient = 'records')
        database=db.list_collection_names()
        print(type(database))
##        for s in range (0,len(database)):
##            if database[s] == filename:
##                war="Please Change File Name"
##                return render_template('landing.html',war=war)
##            else:
        collection1=db[f.filename]
        collection1.insert_many(records_)
        msgs='Upload Successfully'      
                
        return render_template('landing.html',msgs=msgs,ok='ok1',name=f.filename)
    else:
        error = 'Invalid Request'  
        return render_template('landing.html',error=error)

        


@app.route('/download', methods=['GET', 'POST'])
def download():
    results=[]
    if request.method == 'GET':
        return render_template('landing.html',ok='run')
    if request.method == 'POST':
        filename=request.form['filename']
        #print(filename)
        strfile=str(filename)
        cursor = db[strfile].find({},{'_id': False})
        for document in cursor:
            print(document)
            results.append(dict(document))
            fieldnames = [key for key in results[0].keys()]
        return render_template('home.html',msgs='Download Successfully', results=results, fieldnames=fieldnames, len=len)
        
            
if (__name__ == '__main__'):
    app.secret_key="abc123"
    app.run(host='0.0.0.0',port=5000)
