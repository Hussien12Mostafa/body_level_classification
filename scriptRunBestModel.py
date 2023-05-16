import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import sys

# get the command-line arguments
data = sys.argv[1]
df = pd.read_csv(data)

col=['Age', 'Height', 'Weight', 'Veg_Consump', 'Water_Consump', 'Meal_Count',
       'Phys_Act', 'Time_E_Dev', 'Gender_Female', 'Gender_Male',
       'H_Cal_Consump_no', 'H_Cal_Consump_yes', 'Alcohol_Consump_Frequently',
       'Alcohol_Consump_Sometimes', 'Alcohol_Consump_no', 'Smoking_no',
       'Smoking_yes', 'Food_Between_Meals_Always',
       'Food_Between_Meals_Frequently', 'Food_Between_Meals_Sometimes',
       'Food_Between_Meals_no', 'Fam_Hist_no', 'Fam_Hist_yes', 'H_Cal_Burn_no',
       'H_Cal_Burn_yes', 'Transport_Automobile', 'Transport_Bike',
       'Transport_Motorbike', 'Transport_Public_Transportation',
       'Transport_Walking']

df_new = pd.DataFrame()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
x_val1=pd.get_dummies(df)
numeric_var = {"Body Level 1":0, "Body Level 2":1, "Body Level 3":2, "Body Level 4":3}


for i in col:
    if i in x_val1:
        df_new.insert(loc=len(df_new.columns), column=i, value=x_val1[i])
    if i not in x_val1:
        df_new.insert(loc=len(df_new.columns), column=i, value=0)
y_pred = model.predict(df_new)


with open('output.txt', 'w') as file:

    for i in y_pred:
        if i==0:
            file.write('body level 1\n')
        if i==1:
            file.write('body level 2\n') 
        if i==2:
            file.write('body level 3\n')
        if i==3:
            file.write('body level 4\n')
        