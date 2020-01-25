from flask import Flask, render_template,request
import json
import numpy as np
import sys
import pickle
import pandas as pd
from sklearn import preprocessing
app = Flask(__name__)

data=pd.read_csv("C:/Users/diu/Desktop/finalv12.csv")
idList=data['STUDENT_ID'].tolist()
print(len(idList))
data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'STUDENT_ID','STUDENT_NAME'],axis=1, inplace=True)
cat_col=['GRADE_OBTAINED','COURSE_ID','MENTORSFEEDBAC','SEX','RELIGION','MARITAL_STATUS','LOCALGUARDIANRELATION','BEAREDUEXPENSE','PRE_ADDRESS','DROPOUT_BY_PROGRAM']
encoded_data = pd.get_dummies(data,columns=cat_col, drop_first=True)
x=encoded_data.iloc[:,:-1].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
loaded_model=pickle.load(open('C:/Users/diu/Desktop/Environment/DropoutDemoFlask/static/logistic_regr_model.pkl','rb'))
new_prediction = loaded_model.predict_proba(x_scaled)


green = []
yellow = []
red = []

predictionList = {}
colorProbabilityList=[]
k=0
for i in new_prediction:
	j= i[1]
	if j < .40 : 
		green.append('{:.2f}'.format(j).rstrip("0"))
		colorProbabilityList.append('{:.2f}'.format(j).rstrip("0"))
		colorProbabilityList.append('Green')
		predictionList[idList[k]] = colorProbabilityList
		colorProbabilityList=[]
	elif j > .65 : 
		red.append('{:.2f}'.format(j).rstrip("0"))
		colorProbabilityList.append('{:.2f}'.format(j).rstrip("0"))
		colorProbabilityList.append('Red')
		predictionList[idList[k]] = colorProbabilityList
		colorProbabilityList=[]
	else:
		yellow.append('{:.2f}'.format(j).rstrip("0"))
		colorProbabilityList.append('{:.2f}'.format(j).rstrip("0"))
		colorProbabilityList.append('Yellow')
		predictionList[idList[k]] = colorProbabilityList
		colorProbabilityList=[]
	k=k+1
		
color = {
	'red': red,
	'green': green,
	'yellow':yellow
}
print(predictionList)
print(new_prediction)
faculties= [
	{
		'faculty1':'Faculty of Business and Entrepreneurship',
		'departments':
			{
				'department1':'BA',
				'department2':'BS',
				'department3':'RE',
				'department4':'THM',
				'department5':'IE'
				
			}
	},
	{
		'faculty2':'Faculty of Science and Information Technology',
		'departments':
			{
				'department1':'CSE',
				'department2':'CIS',
				'department3':'SWE',
				'department4':'ESDM',
				'department5':'MCT',
				'department6':'GED'
			}
	},
	{
		'faculty3':'Faculty of Engineering',
		'departments':
			{
				'department1':'EEE',
				'department2':'ETE',
				'department4':'TE',
				'department5':'CE'
			}
	},
	{
		'faculty4':'Faculty of Applied Health Sciences',
		'departments':
			{
				'department1':'PHA',
				'department2':'PH',
				'department3':'NFE'
			}
	},
	{
		'faculty5':'Faculty of Humanities and Social Science',
		'departments':
			{
				'department1':'ENG',
				'department2':'JMC',
				'department3':'DS',
				'department4':'ISLM'
			}
	}
]

@app.route('/whole_dashboard')
def Whole_DIU_Deshboard():
	Whole_DashDic={
		'head_title':'Whole Dashboard',
		'title':"Summary of Whole DIU"
	}
	return render_template("Whole_DIU_Dashboard.html",Whole_DashDic=Whole_DashDic,predictionlist=predictionList, faculties=faculties,color=color)

@app.route('/faculties',methods = ['GET','POST'])
def Faculties():
	if request.method == 'POST':
		result = request.form.get('submit')
		if result in "Faculty of Business and Entrepreneurship":
			f=faculties[0]
		elif result in "Faculty of Science and Information Technology":
			f=faculties[1]
		elif result in "Faculty of Engineering":
			f=faculties[2]
		elif result in "Faculty of Applied Health Sciences":
			f=faculties[3]
		elif result in "Faculty of Humanities and Social Science":
			f=faculties[4]
		return render_template("Faculties.html",faculty_title=result,faculties=f,predictionList=predictionList,color=color)
	else :
		return "Sorry"

#Departments
@app.route('/departments',methods = ['GET','POST'])
def Departments():
	if request.method == 'POST':
		result = request.form.get('submit')
		return render_template("Departments.html",department_title=result,predictionlist=predictionList,color=color,j=0)
	else :
		return "Sorry"

@app.route('/Student',methods = ['GET','POST'])
def Student():
	if request.method == 'POST':
		studentId = request.form.get('submit')
		prob = request.form.get('prob')
		prob = float(prob)*100
		return render_template("Student.html",studentId=studentId,prob=prob)
	else :
		return "Sorry"
	



if __name__ == "__main__":
    app.run(debug=True)



