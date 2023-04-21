from flask import Flask, request, jsonify, render_template
import joblib
import re
import pandas as pd
import datetime as dt
from lightgbm import LGBMRegressor
import numpy as np
from dateutil import tz

app = Flask(__name__)

def load_model(x):
    filename = "lgbmr_forecast"    
    if x == "Kabel Opstyg / SKTR":
        x = "Kabel Opstyg SKTR" 
    x = re.sub('[^A-Za-z0-9_]+', '_', x)
    model = joblib.load(f"{filename}_{x}.joblib")
    
    return model

def onehotencode(df, categorical_column):
    column_name_list = list()
    all_category_list = list()

    for column in categorical_column:
        categories = df[column].unique().tolist()

        for category in categories:
            this_list = ((df[column]==category) * 1).tolist()

            all_category_list.append(this_list)
            column_name_list.append(str(column)+"_"+str(category))

    onehotencode_df = pd.DataFrame(all_category_list).transpose()
    onehotencode_df.columns = column_name_list

    return onehotencode_df

    
def preprocess(list_date, list_posko):    
    dayoftheweek = list()
    dayoftheyear = list()
    monthoftheyear = list()
    kode_posko = list()
    year = list()
    df = pd.DataFrame()

    real_date = list()
    df_show = pd.DataFrame()    

    for x in range(len(list_date)):
        # thisdate = df.loc[x, "Tanggal Lapor"]
        for y in range(1, len(list_posko)+1):
            thisdate = list_date[x]
            # print(thisdate)
            thisdayoftheweek = dt.datetime.strptime(str(thisdate), "%Y-%m-%d").strftime('%A')
            thisdayoftheyear = dt.datetime.strptime(str(thisdate), "%Y-%m-%d").strftime('%j')
            thismonthoftheyear = dt.datetime.strptime(str(thisdate), "%Y-%m-%d").strftime('%B')
            thisyear = dt.datetime.strptime(str(thisdate), "%Y-%m-%d").strftime('%Y')

            dayoftheweek.append(thisdayoftheweek)
            dayoftheyear.append(thisdayoftheyear)
            monthoftheyear.append(thismonthoftheyear)
            year.append(thisyear)
            kode_posko.append(y)
            real_date.append(thisdate)

    df["DayOfTheWeek"] = dayoftheweek
    df["DayOfTheYear"] = dayoftheyear
    df["MonthOfTheYear"] = monthoftheyear
    df["Year"] = year
    df["kode_posko"] = kode_posko

    df_show["Date"] = real_date
    df_show["Posko"] = kode_posko

    dayoftheweek_scale_dict = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                                'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    monthoftheyear_scale_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                                'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9,
                                'October': 10, 'November': 11, 'December': 12}
    posko_code_dict = {1: 'POSKO ULP KALEBAJENG', 2: 'POSKO ULP SUNGGUMINASA', 3: 'POSKO ULP TAKALAR', 
                        4: 'POSKO ULP PANAKKUKANG', 5: 'POSKO ULP MATTOANGING', 6: 'POSKO ULP MALINO' }
    # curah_hujan_scale_dict = {'berawan': 1, 'ringan': 2, 'sedang': 3, 'lebat': 4, 'sangat lebat': 5, 
    #                           'ekstreem': 6}


    df["dayoftheweek_scale"] = df["DayOfTheWeek"].map(dayoftheweek_scale_dict)
    df["monthoftheyear_scale"] = df["MonthOfTheYear"].map(monthoftheyear_scale_dict)
    df_show["Posko"] = df["kode_posko"].map(posko_code_dict)
    # df["curah_hujan_scale"] = df["KLASIFIKASI CURAH HUJAN"].map(curah_hujan_scale_dict)

    day_categorical = ["DayOfTheWeek"]
    onehotencode_df = onehotencode(df, day_categorical)
    column_to_drop = ["DayOfTheWeek", "MonthOfTheYear"]
    df.drop(column_to_drop, axis=1, inplace=True)
    df[["DayOfTheYear", "Year"]] = df[["DayOfTheYear", "Year"]].apply(pd.to_numeric, errors="coerce")
    

    df = pd.concat([df, onehotencode_df], axis=1)    
    train_columns = ['DayOfTheYear', 'Year', 'dayoftheweek_scale', 'monthoftheyear_scale',
       'kode_posko', 'DayOfTheWeek_Wednesday', 'DayOfTheWeek_Thursday',
       'DayOfTheWeek_Friday', 'DayOfTheWeek_Saturday', 'DayOfTheWeek_Sunday',
       'DayOfTheWeek_Monday', 'DayOfTheWeek_Tuesday']
    df = df[train_columns]

    return df, df_show    

@app.route("/")
def Home():
    return render_template("home.html")
    
@app.route("/forecast", methods=["GET"])
def forecast_this():
    list_laporan_gangguan = ["Periksa Meter", "Informasi Salah", "APP", "Kabel SR", "Pelanggan", "IML", "Kabel Opstyg / SKTR", "PHB TR", "Kabel JTR", 
                 "Drop Tegangan", "Konstruksi", "MV Cell", "Pemeliharaan", "Tiang Miring", "Tiang TR", "JTR", "Operasi", "Sambungan Tenaga Listrik dan APP",
                 "Jumlah Gangguan Sistem"]
    df_show = pd.DataFrame()
    datelist = pd.date_range(dt.date.today(), periods=30).strftime("%Y-%m-%d").tolist()
    # datelist = pd.date_range(start="2022-11-01", end="2022-11-30").strftime("%Y-%m-%d").tolist()    
    # datelist = pd.to_datetime(datelist, format="%Y-%m-%d")
    list_posko = ["POSKO ULP KALEBAJENG", "POSKO ULP SUNGGUMINASA", "POSKO ULP TAKALAR",
                  "POSKO ULP PANAKUKKANG", "POSKO ULP MATTOANGING", "POSKO ULP MALINO"]
    # print(datelist)
    test_x, df_show = preprocess(datelist, list_posko)
    # print(test_x["DayOfTheYear"])
    # print(test_x)
    # print(test_x.columns)
    for gangguan in list_laporan_gangguan:
        model = load_model(gangguan)
        y_pred = model.predict(test_x)
        y_pred = np.expm1(y_pred)
        df_show[gangguan] = y_pred

    # print(df_show)
    return render_template('forecast.html', tables=[df_show.to_html(classes="data", header="true")])

@app.route("/test")
def test():
    return pd.date_range(dt.date.today(), periods=30).strftime("%d/%m/%Y %I:%M:%S").tolist()
# def html_table():
# predict(list_laporan_gangguan)
if __name__ == '__main__':    
    app.run(debug=True)
