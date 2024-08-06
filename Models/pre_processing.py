import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("Fuel_Consumption_2000-2022.csv")
data = data.drop(labels=[x for x in range(len(data)) \
            if not(data["FUEL"].iloc[x]=="X" or data["FUEL"].iloc[x]=="Z")], axis=0)

data['EMISSIONS']= data['EMISSIONS'].apply ( lambda x : 1 if x>251 else 0)

q1 = np.percentile(data['FUEL CONSUMPTION'], 25)
q3 = np.percentile(data['FUEL CONSUMPTION'], 75)
iqr = q3 - q1
k = 1.5
lower_bound = q1 - k * iqr
upper_bound = q3 + k * iqr
outliers_fuel = data[(data['FUEL CONSUMPTION'] < lower_bound) | (data['FUEL CONSUMPTION'] > upper_bound)]

# Remove the outliers from the original dataframe
data = data.drop(outliers_fuel.index)

ordinal_encoder = OrdinalEncoder()
dataset = data
dataset["FUEL"]=ordinal_encoder.fit_transform(dataset[["FUEL"]])
make_enc = ordinal_encoder.fit_transform(dataset[["MAKE"]])
model_enc = ordinal_encoder.fit_transform(dataset[["MODEL"]])
class_enc = ordinal_encoder.fit_transform(dataset[["VEHICLE CLASS"]])
transmission_enc = ordinal_encoder.fit_transform(dataset[["TRANSMISSION"]])

new_data = np.concatenate((make_enc,model_enc,\
                          class_enc,transmission_enc,\
                          np.asarray(dataset.drop\
                                     (columns =["MAKE","MODEL","VEHICLE CLASS",\
                                                "TRANSMISSION", "EMISSIONS","FUEL CONSUMPTION"]))),axis=1)
train_x,test_x,train_y,test_y = train_test_split(new_data,dataset["EMISSIONS"], test_size=0.1, random_state=42)
