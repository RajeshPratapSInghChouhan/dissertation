#=========================== import ====================================
import os
import pandas as pd
import numpy as np
from pylab import rcParams
import matplotlib.pyplot as pl
import seaborn as sns
import datetime as dt
from PIL import Image, ImageDraw

#=========================== import for web pages ==========================
from flask import Flask, request, render_template
import time

#=============================== ignore warnings ==============
import warnings
warnings.filterwarnings('ignore')

#=========================== PredictPrice ==================================
def PredictPrice(date):

    print ('***************** Start PredictPrice ***********************')
    #============================= input csv =============================
    print (date)
    data = pd.read_csv('./static/database.csv')
    
    #=========================== convert string to time ==================

    data['Date'] = pd.to_datetime(data['Date'])

    #============================= sort by date ==========================

    data = data.sort_values(by=['Date'])

    #=============================== Price Distribution ==================

    pl.figure(figsize=(12,6))
    pl.title("Price Distribution")
    ax = sns.distplot(data["Price"], color = 'b')
    pl.savefig('./static/Price_Distribution.png')

    #=============================== Date v.s. Price ==========================

    pl.figure(figsize=(16,8))
    pl.title("Date v.s. Price")
    ax = sns.tsplot(data["Price"], color = 'b')
    pl.savefig('./static/Date_VS_Price.png')

    #=============================== targate price predicting date ===========

    pred_date = pd.to_datetime(date)

    #====================================== Encoding predicting date =========

    pred_date_code = pred_date.toordinal()

    #====================================== Encoding all date =================

    data.Date = data.Date.map(dt.datetime.toordinal)

    #===================================== data split =======================

    from sklearn.Model_selection import train_test_split

    x = data.drop(['Price'], axis = 1)
    y = data.Price

    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

    #==================================== LinearRegression ======================

    from sklearn.linear_Model import LinearRegression

    linreg =  LinearRegression().fit(x_train,y_train)
    print("")
    print("Linear Regression train data score:{:.3f}".format(linreg.score(x_train,y_train)))
    print("Linear Regression test data score:{:.3f}".format(linreg.score(x_test,y_test)))

    linreg_train_score = round(linreg.score(x_train,y_train), 4)
    linreg_test_score  = round(linreg.score(x_test,y_test), 4)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (linreg_train_score) + '%' , fill=(255,255,0))
    img.save('./static/Linear_Regression_Result_train.png')

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (linreg_test_score) + '%' , fill=(255,255,0))
    img.save('./static/Linear_Regression_Result_test.png')

    #================================================================================

    y_train_int = y_train.astype('int64')
    y_test_int = y_test.astype('int64')

    #===================================== LogisticRegression ===========================

    from sklearn.linear_Model import LogisticRegression

    logreg =  LogisticRegression(penalty='l1', tol=0.0001).fit(x_train,y_train_int)
    print("Logistic Regression train data score:{:.3f}".format(logreg.score(x_train,y_train_int)))
    print("Logistic Regression test data score:{:.3f}".format(logreg.score(x_test,y_test_int)))

    logreg_train_score = round(logreg.score(x_train,y_train_int), 4)
    logreg_test_score  = round(logreg.score(x_test,y_test_int), 4)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (linreg_train_score) + '%' , fill=(255,255,0))
    img.save('./static/Logistic_Regression_Result_train.png')

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (linreg_test_score) + '%' , fill=(255,255,0))
    img.save('./static/Logistic_Regression_Result_test.png')

    #==================================== xgboost ============================================

    import xgboost
    from sklearn.metrics import explained_variance_score

    # XGBoost Regressor
    xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.1, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=8)
    # fit data
    xgb.fit(x_train,y_train)

    #================================================================================

    predictions = xgb.predict(x_test)

    print(explained_variance_score(predictions,y_test))

    xgb_train_score = round(explained_variance_score(predictions,y_test), 4)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (xgb_train_score) + '%' , fill=(255,255,0))
    img.save('./static/XGBoost_Regression_Result_test.png')

    #================================================================================
    
    from sklearn.svm import SVR
    
    svr_lin = SVR(kernel= 'linear', C= 1000 , cache_size=15000, shrinking=True).fit(x_train,y_train)
    print("SVR Linear Model train data score:{:.3f}".format(svr_lin.score(x_train,y_train)))
    print("SVR Linear Model test data score:{:.3f}".format(svr_lin.score(x_test,y_test)))

    svr_lin_train_score = round(svr_lin.score(x_train,y_train), 4)
    svr_lin_test_score  = round(svr_lin.score(x_test,y_test), 4)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (svr_lin_train_score) + '%' , fill=(255,255,0))
    img.save('./static/SVR_Linear_Regression_Result_train.png')

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (svr_lin_test_score) + '%' , fill=(255,255,0))
    img.save('./static/SVR_Linear_Regression_Result_test.png')

    
    #================================================================================
    
    from sklearn.svm import SVR
    
    svr_poly = SVR(kernel= 'poly', C= 1000, degree= 2, cache_size=15000, shrinking=True).fit(x_train,y_train)
    print("SVR Polynomial Model train data score:{:.3f}".format(svr_poly.score(x_train,y_train)))
    print("SVR Polynomial Model test data score:{:.3f}".format(svr_poly.score(x_test,y_test)))

    svr_poly_train_score = round(svr_poly.score(x_train,y_train), 4)
    svr_poly_test_score  = round(svr_poly.score(x_test,y_test), 4)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (svr_poly_train_score) + '%' , fill=(255,255,0))
    img.save('./static/SVR_Polynomial_Regression_Result_train.png')

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (svr_poly_test_score) + '%' , fill=(255,255,0))
    img.save('./static/SVR_Polynomial_Regression_Result_test.png')
    
    #================================================================================
    
    from sklearn.svm import SVR
    
    svr_rbf =  SVR(kernel= 'rbf', C= 1000, gamma= 0.1, cache_size=15000, shrinking=True).fit(x_train,y_train)
    print("SVR RBF Model train data score:{:.3f}".format(svr_rbf.score(x_train,y_train)))
    print("SVR RBF Model test data score:{:.3f}".format(svr_rbf.score(x_test,y_test)))

    svr_rbf_train_score = round(svr_rbf.score(x_train,y_train), 4)
    svr_rbf_test_score  = round(svr_rbf.score(x_test,y_test), 4)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (svr_rbf_train_score) + '%' , fill=(255,255,0))
    img.save('./static/SVR_RBF_Regression_Result_train.png')

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (svr_rbf_test_score) + '%' , fill=(255,255,0))
    img.save('./static/SVR_RBF_Regression_Result_test.png')
    
    #================================================================================
    
    pl.scatter(x_train, y_train, color= 'black', label= 'Data') # plotting the initial datapoints 
    pl.plot(x_train, svr_rbf.predict(x_train), color= 'red', label= 'RBF Model') # plotting the line made by the RBF kernel
    pl.xlabel('Date')
    pl.ylabel('Price')
    pl.title('Support Vector Regression')
    pl.legend()
    pl.savefig('./static/RBF_Model.png')
    
    #================================================================================
    
    pl.scatter(x_train, y_train, color= 'black', label= 'Data') # plotting the initial datapoints 
    pl.plot(x_train,svr_lin.predict(x_train), color= 'green', label= 'Linear Model') # plotting the line made by linear kernel
    pl.xlabel('Date')
    pl.ylabel('Price')
    pl.title('Support Vector Regression')
    pl.legend()
    pl.savefig('./static/Linear_Model.png')
    
    #================================================================================
    
    pl.scatter(x_train, y_train, color= 'black', label= 'Data') # plotting the initial datapoints 
    pl.plot(x_train,svr_poly.predict(x_train), color= 'blue', label= 'Polynomial Model') # plotting the line made by polynomial kernel
    pl.xlabel('Date')
    pl.ylabel('Price')
    pl.title('Support Vector Regression')
    pl.legend()
    pl.savefig('./static/Polynomial_Model.png')
    
    #================================================================================
    
    pl.scatter(x_train, y_train, color= 'black', label= 'Data') # plotting the initial datapoints 
    pl.plot(x_train, svr_rbf.predict(x_train), color= 'red', label= 'RBF Model') # plotting the line made by the RBF kernel
    pl.plot(x_train,svr_lin.predict(x_train), color= 'green', label= 'Linear Model') # plotting the line made by linear kernel
    pl.plot(x_train,svr_poly.predict(x_train), color= 'blue', label= 'Polynomial Model') # plotting the line made by polynomial kernel
    pl.xlabel('Date')
    pl.ylabel('Price')
    pl.title('Support Vector Regression')
    pl.legend()
    pl.savefig('./static/Support_Vector_Regression.png')
    
    #================================================================================
    
    #print (pred_date)
    #print (pred_date_code)
    pred_date_code_arry = (np.array([pred_date_code]).reshape(-1,1))
    #print (pred_date_code_arry)
    #print (pred_date_code_arry[0])
    
    #================================ svr_lin ================================================
    
    print ('svr_lin')
    predicted_price = svr_lin.predict(pred_date_code_arry)[0]
    print (predicted_price)
    
    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (predicted_price) , fill=(255,255,0))
    img.save('./static/SVR_Linear_Regression_Result.png')

    #=================================== svr_rbf =============================================
    
    print ('svr_rbf')
    predicted_price = svr_rbf.predict(pred_date_code_arry)[0]
    print (predicted_price)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (predicted_price) , fill=(255,255,0))
    img.save('./static/SVR_RBF_Regression_Result.png')
    
    #================================== svr_poly ==============================================
    
    predicted_price = svr_poly.predict(pred_date_code_arry)[0]
    print ('svr_poly')
    print (predicted_price)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (predicted_price) , fill=(255,255,0))
    img.save('./static/SVR_Polynomial_Regression_Result.png')
    
    #============================== linreg ==================================================
    
    print ('linreg')
    predicted_price = linreg.predict(pred_date_code_arry)[0]
    print (predicted_price)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (predicted_price) , fill=(255,255,0))
    img.save('./static/Linear_Regression_Result.png')
    
    #================================ logreg ================================================
    
    print ('logreg')
    predicted_price = logreg.predict(pred_date_code_arry)[0]
    print (predicted_price)

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (predicted_price) , fill=(255,255,0))
    img.save('./static/Logistic_Regression_Result.png')
    
    #=============================== xgb =================================================
    
    print ('xgb')
    xgbdfObj = pd.DataFrame(pred_date_code, columns = ['Date'], index=['0']) 
    xgbdfObj
    
    predicted_price = xgb.predict(xgbdfObj)
    #print (predicted_price)
    print (predicted_price[0])

    img = Image.new('RGB', (100, 30), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10,10), str (predicted_price[0]) , fill=(255,255,0))
    img.save('./static/XGBoost_Regression_Result.png')
    
    #================================================================================

    print ('****************** End PredictPrice ************************')

#=========================== import ============

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/')

    print('remove old files form ')
    print(target)

    folder = target
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    if not os.path.isdir(target):
            os.mkdir(target)

    upload = request.files.getlist("file")[0]
    print(upload)
    print("{} is the file name".format(upload.filename))
    filename = upload.filename
    destination = "/".join([target, 'database.csv'])
    print ("incoming file:", filename)
    print ("Save it to:", destination)
    upload.save(destination)
     
    date = request.form['date']
    PredictPrice(date)

    time.sleep(1)

    return render_template("result.html", image_name=filename)

if __name__ == "__main__":
    app.run(port=8080)
    #app.run(port=4040, debug=True)

