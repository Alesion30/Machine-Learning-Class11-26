import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pandas.plotting import register_matplotlib_converters

def random_forest(df, feature_name, target_name):
	print('Target Prarameter: {}'.format(target_name))

	# Select test sets
	features= np.array(df[feature_name])
	targets= np.array(df[target_name])
	train_features, test_features, train_labels, test_labels= train_test_split(features, targets, test_size= 0.25, random_state= 42)

	# Train model
	rf= RandomForestRegressor(n_estimators = 1000, random_state = 42)
	rf.fit(train_features, train_labels)

	# Make predictions
	predictions= rf.predict(test_features)
	errors= abs(predictions-test_labels)

	# Determine performance accuracy
	mape= (errors/test_labels)*100
	accuracy= 100-np.mean(mape)
	print('Accuracy: {}%.'.format(round(accuracy, 2)))

	# Calculate variable importance
	importances= list(rf.feature_importances_)
	feature_importances= [(feature, round(importance, 2)) for feature, importance in zip(feature_name, importances)]
	feature_importances= sorted(feature_importances, key= lambda x: x[1], reverse= True)
	for pair in feature_importances:
		print('Variable: {:60} Importance: {}'.format(*pair))
	print('')

	# Visualize
	years= features[:, feature_name.index('Year')]
	months= features[:, feature_name.index('Month')]
	days= features[:, feature_name.index('Day')]
	dates= [str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year, month, day in zip(years, months, days)]
	dates= [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
	true_data= pd.DataFrame(data= {'Date': dates, 'Data': targets})

	years= test_features[:, feature_name.index('Year')]
	months= test_features[:, feature_name.index('Month')]
	days= test_features[:, feature_name.index('Day')]
	dates= [str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year, month, day in zip(years, months, days)]
	dates= [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
	predictions_data= pd.DataFrame(data= {'Date': dates, 'Prediction': predictions})

	plt.plot(true_data['Date'], true_data['Data'], 'b-', label= 'Data')
	plt.plot(predictions_data['Date'], predictions_data['Prediction'], 'ro', label= 'Prediction')
	plt.xticks(rotation= '60'); 
	plt.legend()
	plt.xlabel('Date'); plt.ylabel(target_name); plt.title('Actual and Predicted Values');
	plt.tight_layout()
	plt.show()
	
if __name__=="__main__":
	register_matplotlib_converters()

	# Load dataset
	ep= pd.read_csv("data/ele_power_data.csv")
	wd= pd.read_csv("data/wether_data.csv")
	df= pd.concat([wd, ep], axis=1).drop(columns="DATE")

	# Trim dataset
	col= ["Date", "Holiday", "Local pressure", "Sea level pressure", "Precipitation amount", "Average temperature", "Highest temperature", "Lowest Temperature", "Average humidity", "Minimum humidity", "Average wind speed", "Maximum wind speed", "Maximum instantaneous wind speed", "Sunshine hours", "Peak supply capacity[10^5kW]", "Expected maximum power[10^5kW]", "Expected usage[%]", "Actual maximum power[10^5kW]", "Actual maximum power generation time zone"]
	df= pd.DataFrame(df.values, columns=col)
	df= df.replace({'Precipitation amount': {"--": None}}).replace("3.8 )", "3.8").replace("2.3 )", "2.3").replace("1.7 )", "1.7").replace("4.6 )", "4.6").replace("6.2 )", "6.2")
	df['Year']= pd.DatetimeIndex(pd.to_datetime(df['Date'])).year
	df['Month']= pd.DatetimeIndex(pd.to_datetime(df['Date'])).month
	df['Day']= pd.DatetimeIndex(pd.to_datetime(df['Date'])).day
	df.drop('Date', axis= 1)
	df.iloc[:, 1:]= df.iloc[:, 1:].astype(float)

	# Run random forest
	features= ["Year", "Month", "Day", "Local pressure", "Sea level pressure", "Average temperature", "Highest temperature", "Lowest Temperature", "Average humidity", "Minimum humidity", "Average wind speed", "Maximum wind speed", "Maximum instantaneous wind speed", "Sunshine hours"]
	targets= ["Peak supply capacity[10^5kW]", "Expected maximum power[10^5kW]", "Expected usage[%]", "Actual maximum power[10^5kW]", "Actual maximum power generation time zone"]
	for target in targets:
		random_forest(df.copy(), features, target)