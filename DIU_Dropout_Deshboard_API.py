from flask import Flask, render_template,request
from keras.models import model_from_json
import json
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
app = Flask(__name__)

#Extracting the json file
json_file=open('C:/Users/diu/Desktop/Environment/DropoutDemoFlask/static/classifier2.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)

#load weights into new model
loaded_model.load_weights('C:/Users/diu/Desktop/Environment/DropoutDemoFlask/static/classifier2.h5')
print('loaded model from disk')


#compiling the loaded model
loaded_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



#evaluate the loaded model from an example
list = [[1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,1,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,1,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,30,3.43,3.50,3.98,1,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,1,16,7,1,2,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,2.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,1,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,25,3.43,3.50,3.98,1,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,13,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145],
            [1,1,2016,56,4,2,5,12,7,19,3.43,3.50,3.98,24,16,5,1,1,2,100001,0,1998,5,2,106642,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1450,4,3,1,0,145],              
            [1,1,2016,56,4,2,5,10,7,21,3.23,3.50,3.98,24,16,7,1,1,2,100961,0,2000,5,2,10000,1,1999,4.0,1,3,2008,3.56,2,1,2,1,2,1,1,106,3,2,2,2,2,1150,4,3,1,0,145]]

x=np.array(list)

#getting the probability for our example dataset
new_prediction = loaded_model.predict(sc.fit_transform(x))
green = []
yellow = []
red = []

predictionList = {}
colorProbabilityList=[]

idList = ['152-15-6040', '152-15-6041', '152-15-6042', '152-15-6043', '152-15-6044', '152-15-6045', '152-15-6046', '152-15-6047', '152-15-6048', '152-15-6049', '152-15-6050', '152-15-6051', '152-15-6052', '152-15-6053', '152-15-6054', '152-15-6055', '152-15-6056', '152-15-6057', '152-15-6058', '152-15-6059', '152-15-6060', '152-15-6061', '152-15-6062', '152-15-6063', '152-15-6064', '152-15-6065', '152-15-6066', '152-15-6067', '152-15-6068', '152-15-6069', '152-15-6070', '152-15-6071', '152-15-6072']
k=0
for i in new_prediction:
	for j in i:
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
    app.run()



