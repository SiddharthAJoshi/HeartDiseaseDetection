
import tkinter #importing tkinter library for GUI creation
from tkinter import *

import pandas as pnd  # importing pandas data analysis toolkit
import numpy as np    # importing numpy library for array operations
from time import time # importing time library for time calculations
from sklearn.model_selection import train_test_split # importing module model_classification from scikit-learn library

   
header_row = ['age', 'sex', 'pain', 'BP', 'chol', 'fbs', 'ecg', 'maxhr', 'eiang', 'eist', 'slope', 'vessels', 'thal',
              'diagnosis']     # Declaring the header row for getting data from the dataset files


# filter to only those diagnosed with heart disease

master = Tk()         # Defining the Tkinter widget
master.wm_title("Heart Disease Prediction")
import sklearn        # Importing scikit-learn functions

Lab=Label(master,text=" Automatic Heart Disease Detection ")    # Adding Label to the Tkinter widget
Lab.grid(row=1,column=5,columnspan=2)                           # Packing the label data to the tkinter widget in user defined rows and columns
Lab.configure(height=2)                                         # Changing dimensions of the Label

Lab=Label(master,text="")
Lab.grid(row=2,column=5,columnspan=2)

Lab1=Label(master,text="Classification Report")
Lab1.grid(row=11,column=2,columnspan=3)
Lab2=Label(master,text="Confusion Matrix")
Lab2.grid(row=11,column=7,columnspan=3)    
T = Text(master, height=6, width=30)                            # Declaring Text Widget for Result Displaying
T.grid(row=13,column=2,rowspan=4, columnspan=3, sticky= 'nsew')
T1 = Text(master, height=6, width=30)
T1.grid(row=13,column=7,rowspan=4,columnspan=3, sticky= 'nsew')

var = StringVar(master)
var.set("Select Dataset") # initial value

option = OptionMenu(master, var, "Cleveland", "Hungarian", "VA", "all") # Declaring the OptionMenu (Drop-Down list) widget
option.grid(row=2,column=5,sticky='nsew')                               # N-North (Top), S-South (Bottom), E-East (Right), W-West (Left)
option.configure(width=6)

field1="Age"                                                            # Defining the field names which user has to input for heart disease detection
field2="Sex"
field3="Pain"
field4="BP"
field5="Chol"
field6="FBS"
field7="ECG"
field8="Maxhr"
field9="Eiang"
field10="Eist"
field11="Slope" 
field12="Vessels"
field13="Thal"


L1=Label(master,text=field1)
L1.grid(row = 4, column = 0, sticky='nsew')
L1.configure(width=14)
L2=Label(master,text=field2)
L2.grid(row = 4, column = 1, sticky='nsew')
L2.configure(width=14)
L3=Label(master,text=field3)
L3.grid(row = 4, column = 2, sticky='nsew')
L3.configure(width=14)
L4=Label(master,text=field4)
L4.grid(row = 4, column = 3, sticky='nsew')
L4.configure(width=14)
L5=Label(master,text=field5)
L5.grid(row = 4, column = 4, sticky='nsew')
L5.configure(width=14)
L6=Label(master,text=field6, )
L6.grid(row = 4, column = 5, sticky='nsew')
L6.configure(width=14)
L7=Label(master,text=field7)
L7.grid(row = 4, column = 6, sticky='nsew')
L7.configure(width=14)
L8=Label(master,text=field8)
L8.grid(row = 4, column = 7, sticky='nsew')
L8.configure(width=14)
L9=Label(master,text=field9)
L9.grid(row = 4, column = 8, sticky='nsew')
L9.configure(width=14)
L10=Label(master,text=field10)
L10.grid(row = 4, column = 9, sticky='nsew')
L10.configure(width=14)
L11=Label(master,text=field11)
L11.grid(row = 4, column = 10, sticky='nsew')
L11.configure(width=14)
L12=Label(master,text=field12)
L12.grid(row = 4, column = 11, sticky='nsew')
L12.configure(width=14)
L13=Label(master,text=field13)
L13.grid(row = 4, column = 12, sticky='nsew')
L13.configure(width=14)

E1=Entry(master)                                                            # Declaring the Entry widget for taking user input values
E1.grid(row = 5, column = 0, sticky='nsew')
E1.configure(width=14)
E2=Entry(master)
E2.grid(row = 5, column = 1, sticky='nsew')
E2.configure(width=14)
E3=Entry(master)
E3.grid(row = 5, column = 2, sticky='nsew')
E3.configure(width=14)
E4=Entry(master)
E4.grid(row = 5, column = 3, sticky='nsew')
E4.configure(width=14)
E5=Entry(master)
E5.grid(row = 5, column = 4, sticky='nsew')
E5.configure(width=14)
E6=Entry(master)
E6.grid(row = 5, column = 5, sticky='nsew')
E6.configure(width=14)
E7=Entry(master)
E7.grid(row = 5, column = 6, sticky='nsew')
E7.configure(width=14)
E8=Entry(master)
E8.grid(row = 5, column = 7, sticky='nsew')
E8.configure(width=14)
E9=Entry(master)
E9.grid(row = 5, column = 8, sticky='nsew')
E9.configure(width=14)
E10=Entry(master)
E10.grid(row = 5, column = 9, sticky='nsew')
E10.configure(width=14)
E11=Entry(master)
E11.grid(row = 5, column = 10, sticky='nsew')
E11.configure(width=14)
E12=Entry(master)
E12.grid(row = 5, column = 11, sticky='nsew')
E12.configure(width=14)
E13=Entry(master)
E13.grid(row = 5, column = 12, sticky='nsew')
E13.configure(width=14)

Labx=Label(master,text="")
Labx.grid(row=21,column=4,columnspan=4)
Labx.visible=False

#T3 = Text(master, height=2, width=30)                                                       # Declaring Text Widget for Displaying Prediction
#T3.grid(row=23,column=4, columnspan=4, sticky= 'nsew')

def train_classifier(x_train,x_test,y_train,y_test,string):                         # Declaring the function for training classifiers and classification analysis
    global clf                                                                      # Declaring clf as a Global Variable for using throughot the code
    global outclass
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    T.delete(1.0,END)                                                               # Deleting the text in the Text Widget
    T1.delete(1.0,END)
    if string=="SVM":
        from sklearn import svm
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        t1= time()
        param_grid = {'C': [1, 5, 10, 50, 100],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)  # Initialization of GridSearch Optimization for SVM with RBF kernel

        clf.fit(x_train, y_train)                                                   # Fitting the classifier to the training and testing the SVM Classifier    
        y_pred = clf.predict(x_test)                                                # Predict Results for Test Data
        title = "Learning Curves (SVM)"
        geterror(x_train,y_train,clf,title);
        t= time()-t1
        print("Training Complete")
        
    elif string=="Naive Bayes":
        from sklearn.naive_bayes import GaussianNB                              
        t2=time()
        clf = GaussianNB()                                                          # Initializing the Naive Bayes Classifier
        clf.partial_fit(x_train, y_train, np.unique(y_train))                       # Fitting the classifier to the training and testing the Naive Bayes Classifier
        y_pred = clf.predict(x_test)
        t = time() - t2
        title = "Learning Curves (Naive Bayes)"
        geterror(x_train,y_train,clf,title);
        
        print("Training Complete")

    elif string=="Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        t3=time()
        clf=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, solver='liblinear', max_iter=100, verbose=0, warm_start=False, n_jobs=1)   # Initializing the Logistic Regression Classifier
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        t = time() - t3
        title = "Learning Curves (Logistic Regression)"
        geterror(x_train,y_train,clf,title);
        print("Training Complete")

    t=str(t)
    
    classre=classification_report(y_test,y_pred)                       # Generating Classification Report
    T.insert(END,classre[1:5]+classre[1:32]+classre[1:13]+classre[60:90]+classre[1:11]+classre[1:2]+classre[115:140]+classre[1:7]+classre[161:195]) # Printing Precision and Recall Results 
    print(classre)
    confmat=confusion_matrix(y_test, y_pred)                                        # Calculating the Confusion Matrix for the classification
    T1.insert(END, confmat)
    T.insert(END, classre[1:9])
    T.insert(END, "Accuracy   ")
    T.insert(END, classre[1:5])
    T.insert(END, int(float((y_test==y_pred).sum())/len(y_test.T)*100))
    T.insert(END, "%")
    T.insert(END, classre[1:10]+classre[1:10])
    T.insert(END, "Class. Time")
    T.insert(END, classre[1:8])
    T.insert(END, t[0:4]+" sec")
    import matplotlib.pyplot as plt
def process_dataset(string):

        
    
    if string=="Cleveland":
        
        heart = pnd.read_csv('processed.cleveland.data', names=header_row)          # Reading the dataset file in .data format using Pandas library function read_csv()
        print("Unprocessed Cleveland Dataset")
        print("************************************************************************")
        print(heart.loc[:, 'age':'diagnosis'])
        print("************************************************************************")

        import numpy as np
        has_hd_check = heart['diagnosis'] > 0                                                           # Getting the indices of individuals having heart disease
        has_hd_patients = heart[has_hd_check]
        heart['vessels'] = heart['vessels'].apply(lambda vessels: 0.0 if vessels == "?" else vessels)   # Replacing the unknown values in the dataset with float
        heart['vessels'] = heart['vessels'].astype(float)
        heart['thal'] = heart['thal'].apply(lambda thal: 0.0 if thal == "?" else thal)
        heart['thal'] = heart['thal'].astype(float)
        heart['diag_int'] = has_hd_check.astype(int)

        ind1 = np.where((heart['diagnosis'] == 1)|(heart['diagnosis'] ==2));
        ind2 = np.where((heart['diagnosis'] == 3)|(heart['diagnosis'] ==4));

        temp = heart['diagnosis'];
        temp.ix[ ind1 ] = 1;
        temp.ix[ ind2 ] = 2;
        heart['diagnosis'] = temp;

        global x_train
        global y_train
        global x_test
        global y_test
        x_train, x_test, y_train, y_test = train_test_split(heart.loc[:, 'age':'thal'], heart.loc[:, 'diagnosis'],   # Splitting the processed data into training data and testing data
                                                        test_size=0.30, random_state=42)                            # test_size = percent of data used for testing,
                                                                                                                    # random_state = for initializing the random number generator

        print("Processed Cleveland Dataset")
        print("************************************************************************")
        print(heart.loc[:, 'age':'diagnosis'])
        print("************************************************************************")
        

    elif string=="VA":

        import numpy as np


        
        heart_va = pnd.read_csv('processed.va.data', names=header_row)
        print("Unprocessed VA Dataset")
        print("************************************************************************")
        print(heart_va.loc[:, 'age':'diagnosis'])
        print("************************************************************************")

        has_hd_check = heart_va['diagnosis'] > 0

        
        
        heart_va['diag_int'] = has_hd_check.astype(int) 
        heart_va = heart_va.replace(to_replace='?', value=0.0)
        heart_va['diag_int'] = has_hd_check.astype(int)
        
        ind1 = np.where((heart_va['diagnosis'] == 1)|(heart_va['diagnosis'] ==2));
        ind2 = np.where((heart_va['diagnosis'] == 3)|(heart_va['diagnosis'] ==4));

        temp = heart_va['diagnosis'];
        temp.ix[ ind1 ] = 1;
        temp.ix[ ind2 ] = 2;

        heart_va['diagnosis'] = temp;
        
        print("Processed VA Dataset")
        print("************************************************************************")
        print(heart_va.loc[:, 'age':'diagnosis'])
        print("************************************************************************")
         
        x_train, x_test, y_train, y_test = train_test_split(heart_va.loc[:, 'age':'thal'], heart_va.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)
        
    elif string=="Hungarian":
        import numpy as np
        heart_hu = pnd.read_csv('processed.hungarian.data', names=header_row)
        print("Unprocessed Hungarian Dataset")
        print("************************************************************************")
        print(heart_hu.loc[:, 'age':'diagnosis'])
        print("************************************************************************")

        has_hd_check = heart_hu['diagnosis'] > 0
        heart_hu['diag_int'] = has_hd_check.astype(int)
        heart_hu = heart_hu.replace(to_replace='?', value=0.0)

        ind1 = np.where((heart_hu['diagnosis'] == 1)|(heart_hu['diagnosis'] ==2));
        ind2 = np.where((heart_hu['diagnosis'] == 3)|(heart_hu['diagnosis'] ==4));

        temp = heart_hu['diagnosis'];
        temp.ix[ ind1 ] = 1;
        temp.ix[ ind2 ] = 2;
        heart_hu['diagnosis'] = temp;

        print("Processed Hungarian Dataset")
        print("************************************************************************")
        print(heart_hu.loc[:, 'age':'diagnosis'])
        print("************************************************************************")
        heart_hu['diag_int'] = has_hd_check.astype(int)

        
        x_train, x_test, y_train, y_test = train_test_split(heart_hu.loc[:, 'age':'thal'], heart_hu.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)

    elif string=="all":
        import numpy as np
        heart_cl = pnd.read_csv('processed.cleveland.data', names=header_row)
        print("Unprocessed Cleveland Dataset")
        print("************************************************************************")
        print(heart_cl.loc[:, 'age':'diagnosis'])
        print("************************************************************************")
        has_hd_check = heart_cl['diagnosis'] > 0
        has_hd_patients = heart_cl[has_hd_check]
        heart_cl['diag_int'] = has_hd_check.astype(int)
        heart_cl['vessels'] = heart_cl['vessels'].apply(lambda vessels: 0.0 if vessels == "?" else vessels)
        heart_cl['vessels'] = heart_cl['vessels'].astype(float)
        heart_cl['thal'] = heart_cl['thal'].apply(lambda thal: 0.0 if thal == "?" else thal)
        heart_cl['thal'] = heart_cl['thal'].astype(float)

        ind1 = np.where((heart_cl['diagnosis'] == 1)|(heart_cl['diagnosis'] ==2));
        ind2 = np.where((heart_cl['diagnosis'] == 3)|(heart_cl['diagnosis'] ==4));

        temp = heart_cl['diagnosis'];
        temp.ix[ ind1 ] = 1;
        temp.ix[ ind2 ] = 2;
        heart_cl['diagnosis'] = temp;

        heart_va = pnd.read_csv('processed.va.data', names=header_row)
        print("Unprocessed VA Dataset")
        print("************************************************************************")
        print(heart_va.loc[:, 'age':'diagnosis'])
        print("************************************************************************")

        has_hd_check = heart_va['diagnosis'] > 0
        heart_va['diag_int'] = has_hd_check.astype(int)
        heart_va = heart_va.replace(to_replace='?', value=0.0)

        ind1 = np.where((heart_va['diagnosis'] == 1)|(heart_va['diagnosis'] ==2));
        ind2 = np.where((heart_va['diagnosis'] == 3)|(heart_va['diagnosis'] ==4));

        temp = heart_va['diagnosis'];
        temp.ix[ ind1 ] = 1;
        temp.ix[ ind2 ] = 2;
        heart_va['diagnosis'] = temp;

        print("Processed VA Dataset")
        print("************************************************************************")
        print(heart_va.loc[:, 'age':'diagnosis'])
        print("************************************************************************")

        heart_hu = pnd.read_csv('processed.hungarian.data', names=header_row)
        print("Unprocessed Hungarian Dataset")
        print("************************************************************************")
        print(heart_hu.loc[:, 'age':'diagnosis'])
        print("************************************************************************")

        has_hd_check = heart_hu['diagnosis'] > 0
        heart_hu['diag_int'] = has_hd_check.astype(int)
        heart_hu = heart_hu.replace(to_replace='?', value=0.0)

        ind1 = np.where((heart_hu['diagnosis'] == 1)|(heart_hu['diagnosis'] ==2));
        ind2 = np.where((heart_hu['diagnosis'] == 3)|(heart_hu['diagnosis'] ==4));

        temp = heart_hu['diagnosis'];
        temp.ix[ ind1 ] = 1;
        temp.ix[ ind2 ] = 2;
        heart_hu['diagnosis'] = temp;

        print("Processed Hungarian Dataset")
        print("************************************************************************")
        print(heart_hu.loc[:, 'age':'diagnosis'])
        print("************************************************************************")

        x_train1, x_test1, y_train1, y_test1 = train_test_split(heart_cl.loc[:, 'age':'thal'], heart_cl.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)
        x_train2, x_test2, y_train2, y_test2 = train_test_split(heart_va.loc[:, 'age':'thal'], heart_va.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)
        x_train3, x_test3, y_train3, y_test3 = train_test_split(heart_hu.loc[:, 'age':'thal'], heart_hu.loc[:, 'diagnosis'],
                                                        test_size=0.30, random_state=42)

        # Combining the dataset for Cleveland, VA and Hungarian Dataset
        x_train4= x_train1.append(x_train2);
        x_train = x_train4.append(x_train3);
        
        y_train4 = y_train1.append(y_train2);
        y_train = y_train4.append(y_train3);

        x_test4 = x_test1.append(x_test2);
        x_test = x_test4.append(x_test3)

        y_test4 = y_test1.append(y_test2);
        y_test = y_test4.append(y_test3);
        

button = Button(master, text="Process Dataset", command=lambda: process_dataset(var.get())) #Defining the button in the Tkinter Widget
button.grid(row=2, column=6, sticky='nsew')
button.configure(width=14)
var1 = StringVar(master)
var1.set("Select Classifier") # initial value

option1 = OptionMenu(master, var1, "SVM", "Naive Bayes", "Logistic Regression")
option1.grid(row=3,column=5, sticky='nsew')
option1.configure(width=14)
#option.place ( relx=0.5, rely=0.1)
button1 = Button(master, text=" Train Classifier", command=lambda: train_classifier(x_train,x_test,y_train,y_test,var1.get()))
button1.grid(row=3,column=6, sticky='nsew')
button1.configure(width=14)




#e1.bind('<Button-1>',e1.delete(0,END))

def predres(clf):                                                                           # Defining function to predict the result from the user input data

    
    E14=E10.get()                                                                           # Converting the Eist data according to sign and decimal point
    if len(E14)==3 or len(E14)==4:
        E15=float(E14)
    else:
        E16=E14+'.0'
        E15=float(E16)
        
    test=[float(E1.get()+'.0'),float(E2.get()+'.0'),float(E3.get()+'.0'),float(E4.get()+'.0'),float(E5.get()+'.0'),float(E6.get()+'.0'),float(E7.get()+'.0'),float(E8.get()+'.0'),float(E9.get()+'.0'),E15,float(E11.get()+'.0'),float(E12.get()+'.0'),float(E13.get()+'.0')]
    test=np.reshape(test,(1,-1))
    if clf.predict(test) == 1:
        
        Labx1=Label(master,text="The Person has Mild Heart Disease", bg='orange')
        Labx1.visible=False
        Labx1.grid(row=23,column=4,columnspan=4, sticky = 'nsew')
        Labx1.visible=True
        #T3.insert(END,"The Person has Heart Disease")

    elif clf.predict(test) == 2:
        Labx1=Label(master,text="The Person has Severe Heart Disease", bg='red')
        Labx1.visible=False
        Labx1.grid(row=23,column=4,columnspan=4, sticky = 'nsew')
        Labx1.visible=True
        
    else:
        Labx1=Label(master,text="The Person does not have Heart Disease", bg='green')
        Labx1.visible=False
        Labx1.grid(row=23,column=4,columnspan=4, sticky = 'nsew')
        Labx1.visible=True
        #T3.insert(END,"The Person does not have Heart Disease")
        

def geterror(x_train,y_train,clf,title):
    #global outclass
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import ShuffleSplit
    clas=[];
    
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        #plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plt.axis([ 0,len(y),0,1.1])
        return plt

    
    for index in range(len(y_train)):
        x_train1=np.reshape(x_train.iloc[index,:],(1,-1));
        outclass = clf.predict(x_train1);
        clas.append(outclass[0]);
        
    ind3 = np.where((clas == y_train));
    ind4 = np.where((clas != y_train));
    l=0
    m=0
    
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=0)
    
    plot_learning_curve(clf, title, x_train, y_train, (0.7, 1.01), cv=cv, n_jobs=1)
    
    plt.show()
    
    
button2 = Button(master, text=" Predict Heart Disease ", command=lambda:predres(clf))
button2.grid(row=8,column=5,columnspan=2,sticky='we')
button2.configure(width=14)
#button1.place(relx=0.1,rely=0.2)

master.mainloop()                                                                           # To continuously show the Tkinter widget until manually closed
