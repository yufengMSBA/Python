from tkinter import *
import numpy as np
import pandas as pd
from tkinter.scrolledtext import *
import matplotlib.pyplot as plt
import io
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from IPython import get_ipython
import random
import time
import threading
import os

buf = io.StringIO()
def run1():
    global filecd
    filedirctory=str(cdinput.get())
    filename=str(nameinput.get())
    filecd=filedirctory+'/'+filename
    return(filecd)

def runf2():
    heloc = pd.read_csv(filecd,engine='python')
    df = pd.DataFrame(heloc)
    mymap = {'Good':1, 'Bad':0}
    df = df.applymap(lambda s: mymap.get(s) if s in mymap else s)
    without9_filter = \
    (df['ExternalRiskEstimate']!=-9) & \
    (df['MSinceOldestTradeOpen']!=-9) & \
    (df['MSinceMostRecentTradeOpen']!=-9) & \
    (df['AverageMInFile']!=-9) & \
    (df['NumSatisfactoryTrades']!=-9) & \
    (df['NumTrades60Ever2DerogPubRec']!=-9) & \
    (df['NumTrades90Ever2DerogPubRec']!=-9) & \
    (df['PercentTradesNeverDelq']!=-9) & \
    (df['MSinceMostRecentDelq']!=-9) & \
    (df['MaxDelq2PublicRecLast12M']!=-9) & \
    (df['MaxDelqEver']!=-9) & \
    (df['NumTotalTrades']!=-9) & \
    (df['NumTradesOpeninLast12M']!=-9) & \
    (df['PercentInstallTrades']!=-9) & \
    (df['MSinceMostRecentInqexcl7days']!=-9) & \
    (df['NumInqLast6M']!=-9) & \
    (df['NumInqLast6Mexcl7days']!=-9) & \
    (df['NetFractionRevolvingBurden']!=-9) & \
    (df['NetFractionInstallBurden']!=-9) & \
    (df['NumRevolvingTradesWBalance']!=-9) & \
    (df['NumInstallTradesWBalance']!=-9) & \
    (df['NumBank2NatlTradesWHighUtilization']!=-9) & \
    (df['PercentTradesWBalance']!=-9) 
    dfnew = df[without9_filter]
    dfwithnan=dfnew.replace([-8,-7],np.nan)
    dfwithnan.info(buf=buf) 
    txt.insert(END,buf.getvalue())
    data_new1 = dfwithnan.drop("MSinceMostRecentDelq", axis=1)
    data_new2 = dfwithnan.drop("MSinceMostRecentInqexcl7days", axis=1)
    data_new3 = dfwithnan.drop("NetFractionInstallBurden", axis=1)
    data_new=data_new3
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data_new, test_size=0.2, random_state=1)
    data_train_onlyx=train_set.drop("RiskPerformance",axis=1)  
    from sklearn.preprocessing import Imputer
    imputer=Imputer(strategy="median")
    imputer.fit(data_train_onlyx)
    X=imputer.transform(data_train_onlyx)
    data_tr=pd.DataFrame(X,columns=data_train_onlyx.columns)
    X=data_tr[['ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen','AverageMInFile',    
                                 'NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec', 
                                 'PercentTradesNeverDelq','NumTotalTrades','NumTradesOpeninLast12M',
                                 'PercentInstallTrades','NumInqLast6M','NumInqLast6Mexcl7days',                 
                                 'NetFractionRevolvingBurden','NumRevolvingTradesWBalance','NumInstallTradesWBalance',              
                                 'NumBank2NatlTradesWHighUtilization','PercentTradesWBalance',            
                                 'MaxDelq2PublicRecLast12M','MaxDelqEver']]
    Y=train_set['RiskPerformance']
    random.seed(36)
    Bagging_max = []
    RF_max = []
    SingleTree_max = []
    n_Bagging = []
    n_RF = []
    n_SingleTree = []
    df = pd.DataFrame()

    for tree_depth in range(1,9,1):
        clf_tree = tree.DecisionTreeClassifier(max_depth=tree_depth)
        clf_tree = clf_tree.fit(X, Y)    
        clf_tree_scores = cross_val_score(clf_tree, X, Y, cv=5)
        base_clf = tree.DecisionTreeClassifier(max_depth=tree_depth) # base classifier
        results = []
        n_range = range(1,30,1)
        for n in n_range:
            # bagging classifier with n bootstrapped data sets
            clf_bagging = BaggingClassifier(n_estimators=n, base_estimator=base_clf)
            scores = cross_val_score(clf_bagging, X, Y, cv=5)
        
            # random forest classifier with n bootstrapped data sets m=sqrt(p)
            clf_rf = RandomForestClassifier(max_features="sqrt", n_estimators=n, max_depth=tree_depth)
            clf_rf_scores = cross_val_score(clf_rf, X, Y, cv=5)

            results.append((n,scores.mean(), scores.std(),clf_rf_scores.mean(),clf_rf_scores.std(),clf_tree_scores.mean()))
        
        df_accuracy = pd.DataFrame(data=results,columns=['n','Bagging accuracy','Bagging error','RF accuracy','RF error','Single tree'])
        df_accuracy.index=df_accuracy['n']
        df_accuracy = df_accuracy[['Bagging accuracy','RF accuracy','Single tree']]
        Bagging_max.append(max(df_accuracy['Bagging accuracy']))
        RF_max.append(max(df_accuracy['RF accuracy']))
        SingleTree_max.append(max(df_accuracy['Single tree']))
        n_Bagging.append(df_accuracy.idxmax()['Bagging accuracy'])
        n_RF.append(df_accuracy.idxmax()['RF accuracy'])
        n_SingleTree.append(df_accuracy.idxmax()['Single tree'])
    df = pd.DataFrame()
    df['RF'] = RF_max
    df['Rn'] = n_RF 
    df['Bagging'] = Bagging_max
    df['Bn'] = n_Bagging
    df['CART'] = SingleTree_max
    Tree_Depth = df.index+1
    df['Depth'] = Tree_Depth
    modelresult.insert(END,str(df))

def run2(func,*args):
    t=threading.Thread(target=func,args=args)
    t.setDaemon(True)
    t.start()




def run3():
    global newinput
    newinput=pd.DataFrame()
    newinput['ExternalRiskEstimate'] = [float(inp1.get())]
    newinput['MSinceOldestTradeOpen'] = [float(inp2.get())]
    newinput['MSinceMostRecentTradeOpen'] = [float(inp3.get())]
    newinput['AverageMInFile']=[float(inp4.get())]
    newinput['NumSatisfactoryTrades'] = [float(inp5.get())]
    newinput['NumTrades60Ever2DerogPubRec'] = [float(inp6.get())]
    newinput['NumTrades90Ever2DerogPubRec'] = [float(inp7.get())]
    newinput['PercentTradesNeverDelq'] =[float(inp8.get())]
    newinput['MaxDelq2PublicRecLast12M'] =[float(inp9.get())]
    newinput['MaxDelqEver']=[float(inp10.get())]
    newinput['NumTotalTrades']=[float(inp11.get())]
    newinput['NumTradesOpeninLast12M']=[float(inp12.get())]
    newinput['PercentInstallTrades']=[float(inp13.get())]
    newinput['NumInqLast6M']=[float(inp14.get())]
    newinput['NumInqLast6Mexcl7days']=[float(inp15.get())]
    newinput['NetFractionRevolvingBurden']=[float(inp16.get())]
    newinput['NumRevolvingTradesWBalance']=[float(inp17.get())]
    newinput['NumInstallTradesWBalance']=[float(inp18.get())]
    newinput['NumBank2NatlTradesWHighUtilization']=[float(inp19.get())]
    newinput['PercentTradesWBalance']=[float(inp20.get())]
    return(newinput)


def run4():
    global modelnumber
    heloc = pd.read_csv(filecd,engine='python')
    df = pd.DataFrame(heloc)
    mymap = {'Good':1, 'Bad':0}
    df = df.applymap(lambda s: mymap.get(s) if s in mymap else s)
    without9_filter = \
    (df['ExternalRiskEstimate']!=-9) & \
    (df['MSinceOldestTradeOpen']!=-9) & \
    (df['MSinceMostRecentTradeOpen']!=-9) & \
    (df['AverageMInFile']!=-9) & \
    (df['NumSatisfactoryTrades']!=-9) & \
    (df['NumTrades60Ever2DerogPubRec']!=-9) & \
    (df['NumTrades90Ever2DerogPubRec']!=-9) & \
    (df['PercentTradesNeverDelq']!=-9) & \
    (df['MSinceMostRecentDelq']!=-9) & \
    (df['MaxDelq2PublicRecLast12M']!=-9) & \
    (df['MaxDelqEver']!=-9) & \
    (df['NumTotalTrades']!=-9) & \
    (df['NumTradesOpeninLast12M']!=-9) & \
    (df['PercentInstallTrades']!=-9) & \
    (df['MSinceMostRecentInqexcl7days']!=-9) & \
    (df['NumInqLast6M']!=-9) & \
    (df['NumInqLast6Mexcl7days']!=-9) & \
    (df['NetFractionRevolvingBurden']!=-9) & \
    (df['NetFractionInstallBurden']!=-9) & \
    (df['NumRevolvingTradesWBalance']!=-9) & \
    (df['NumInstallTradesWBalance']!=-9) & \
    (df['NumBank2NatlTradesWHighUtilization']!=-9) & \
    (df['PercentTradesWBalance']!=-9) 
    dfnew = df[without9_filter]
    dfwithnan=dfnew.replace([-8,-7],np.nan)
    dfwithnan.info(buf=buf) 
    txt.insert(END,buf.getvalue())
    data_new1 = dfwithnan.drop("MSinceMostRecentDelq", axis=1)
    data_new2 = dfwithnan.drop("MSinceMostRecentInqexcl7days", axis=1)
    data_new3 = dfwithnan.drop("NetFractionInstallBurden", axis=1)
    data_new = data_new3
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data_new, test_size=0.2, random_state=1)
    data_train_onlyx=train_set.drop("RiskPerformance",axis=1)  
    from sklearn.preprocessing import Imputer
    imputer=Imputer(strategy="median")
    imputer.fit(data_train_onlyx)
    X=imputer.transform(data_train_onlyx)
    data_tr=pd.DataFrame(X,columns=data_train_onlyx.columns)
    X=data_tr[['ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen','AverageMInFile',    
                                 'NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec', 
                                 'PercentTradesNeverDelq','NumTotalTrades','NumTradesOpeninLast12M',
                                 'PercentInstallTrades','NumInqLast6M','NumInqLast6Mexcl7days',                 
                                 'NetFractionRevolvingBurden','NumRevolvingTradesWBalance','NumInstallTradesWBalance',              
                                 'NumBank2NatlTradesWHighUtilization','PercentTradesWBalance',            
                                 'MaxDelq2PublicRecLast12M','MaxDelqEver']]
    Y=train_set['RiskPerformance']
    b=int(inp42.get())
    c=int(inp43.get())
    modelnumber = RandomForestClassifier(max_features='sqrt', n_estimators=b, max_depth=c)
    modelnumber=modelnumber.fit(X,Y)
    d=modelnumber.predict(newinput)
    if d==1:
        out='Good'
    else:
        out='Bad'
    finalresult.insert(END,out)










root = Tk()
root.geometry('2048x2048')
root.title('Home Equity Line of Credit (HELOC)')
lb = Label(root, text='Home Equity Line of Credit (HELOC) DSS', \
           bg='#d9d9d9', \
           fg='black', \
           font=('Times', 32), \
           width=200, \
           height=1)
lb.pack()
path1=os.path.abspath('.')
photo = PhotoImage(file=path1+"/blurred-bg-2.png")


#step 1
#label
lbstep1=Label(root, text='Step 1: Load Dataset',font=('Times', 25),anchor='w')
lbstep1.place(rely=0.1, relx=0.02, relwidth=0.22, relheight=0.04)
datainputlab=Label(root, text='Dataset Directory:',font=('Times', 20),anchor='w')
datainputlab.place(rely=0.15, relx=0.02, relwidth=0.12, relheight=0.04)
nameinputlab=Label(root, text='Dataset Name:',font=('Times', 20),anchor='w')
nameinputlab.place(rely=0.21,relx=0.02, relwidth=0.1, relheight=0.04)
#input
cdinput = Entry(root)
cdinput.place(rely=0.19, relx=0.02, relwidth=0.25, relheight=0.025)
nameinput = Entry(root)
nameinput.place(rely=0.246, relx=0.02, relwidth=0.25, relheight=0.025)
#buttom
btn1 = Button(root, text='Confirm', command=run1,relief='sunken',activebackground='#f5f5f5',activeforeground='#d9d9d9')
btn1.place(relx=0.02, rely=0.31, relwidth=0.05, relheight=0.03)



#step 2
#label1
lbstep2=Label(root, text='Step 2: Create ML Classifier',font=('Times',25),anchor='w')
lbstep2.place(rely=0.36, relx=0.02, relwidth=0.22, relheight=0.04)
#buttom
btn2 = Button(root, text='Create Classifier', command=lambda :run2(runf2),relief='sunken',activebackground='#f5f5f5',activeforeground='#d9d9d9')
btn2.place(relx=0.02, rely=0.415, relwidth=0.071, relheight=0.03)
#label2 model 5mins
lbstep21=Label(root, text='wait for 10 mins(part 4 have result)',font=('Times',16),anchor='w')
lbstep21.place(rely=0.408, relx=0.097, relwidth=0.22, relheight=0.05)
#label3 model information
lbstep22=Label(root, text='Training Dataset Information',font=('Times',19),anchor='w')
lbstep22.place(rely=0.45, relx=0.02, relwidth=0.26, relheight=0.04)
#result
txt=ScrolledText(root)
txt.place(relx=0.02,rely=0.49,relwidth=0.25, relheight=0.4)



#canvas1
can1=Canvas(root,bg='#d9d9d9')
can1.place(relx=0.277, rely=0.05, relwidth=0.006, relheight=0.9)




#step 3
#label1
lbstep3=Label(root, text='Step 3: Input New Case',font=('Times',25),anchor='w')
lbstep3.place(rely=0.1, relx=0.29, relwidth=0.2, relheight=0.04)



#canvas2
can2=Canvas(root,bg='#d9d9d9')
can2.place(relx=0.73, rely=0.05, relwidth=0.006, relheight=0.9)

#canvas3
can3=Canvas(root,bg='#d9d9d9')
can3.place(relx=0, rely=0.95, relwidth=1, relheight=0.07)
can3.create_text(20,20,text='Copyright ©2019 CIS432 team17 Yufeng Shen',font=('Times', 15),anchor='w',justify = LEFT)
#bottom
btn3 = Button(root, text='Confirm', command=run3,relief='sunken',activebackground='#f5f5f5',activeforeground='#d9d9d9')
btn3.place(relx=0.458, rely=0.106, relwidth=0.04, relheight=0.03)



#step 4
#lable1
lbstep4=Label(root, text='Step 4: Generate Prediction',font=('Times',25),anchor='w')
lbstep4.place(rely=0.1, relx=0.74, relwidth=0.3, relheight=0.04)
#lable2
lbstep42=Label(root, text='Model Performance on Training Data',font=('Times',19),anchor='w')
lbstep42.place(rely=0.16, relx=0.74, relwidth=0.3, relheight=0.03)
#txt1
modelresult=ScrolledText(root)
modelresult.place(relx=0.74,rely=0.2,relwidth=0.26, relheight=0.21)
#lable3
lbstep43=Label(root, text='Use the Best Model',font=('Times',19),anchor='w')
lbstep43.place(rely=0.45, relx=0.74, relwidth=0.3, relheight=0.03)
#input 
inp41=Entry(root)
inp41.place(rely=0.5,relx=0.78, relwidth=0.03, relheight=0.03)
inp42=Entry(root)
inp42.place(rely=0.5,relx=0.86, relwidth=0.03, relheight=0.03)
inp43=Entry(root)
inp43.place(rely=0.5,relx=0.93, relwidth=0.03, relheight=0.03)
#input label
lb41 = Label(root, text='model',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb41.place(rely=0.5, relx=0.74, relwidth=0.03, relheight=0.03)
lb42 = Label(root, text='n',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb42.place(rely=0.5, relx=0.82, relwidth=0.03, relheight=0.03)
lb43 = Label(root, text='Depth',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb43.place(rely=0.5, relx=0.89, relwidth=0.03, relheight=0.03)
#buttom
btn4 = Button(root, text='Confirm', command=run4,relief='sunken',activebackground='#f5f5f5',activeforeground='#d9d9d9')
btn4.place(relx=0.74, rely=0.54, relwidth=0.04, relheight=0.03)
#label4
lbstep44=Label(root, text='New Prdiction',font=('Times',19),anchor='w')
lbstep44.place(rely=0.58, relx=0.74, relwidth=0.3, relheight=0.03)
#txt
finalresult=ScrolledText(root)
finalresult.place(relx=0.74,rely=0.62,relwidth=0.26, relheight=0.21)





#step 2
#input label
inp1 = Entry(root)
inp1.place(rely=0.15, relx=0.45, relwidth=0.05, relheight=0.05)
inp2 = Entry(root)
inp2.place(rely=0.23, relx=0.45, relwidth=0.05, relheight=0.05)
inp3 = Entry(root)
inp3.place(rely=0.31, relx=0.45, relwidth=0.05, relheight=0.05)
inp4 = Entry(root)
inp4.place(rely=0.39, relx=0.45, relwidth=0.05, relheight=0.05)
inp5 = Entry(root)
inp5.place(rely=0.47, relx=0.45, relwidth=0.05, relheight=0.05)
inp6 = Entry(root)
inp6.place(rely=0.55, relx=0.45, relwidth=0.05, relheight=0.05)
inp7 = Entry(root)
inp7.place(rely=0.63, relx=0.45, relwidth=0.05, relheight=0.05)
inp8 = Entry(root)
inp8.place(rely=0.71, relx=0.45, relwidth=0.05, relheight=0.05)
inp9 = Entry(root)
inp9.place(rely=0.79, relx=0.45, relwidth=0.05, relheight=0.05)
inp10 = Entry(root)
inp10.place(rely=0.15, relx=0.67, relwidth=0.05, relheight=0.05)
inp11 = Entry(root)
inp11.place(rely=0.23, relx=0.67, relwidth=0.05, relheight=0.05)
inp12 = Entry(root)
inp12.place(rely=0.31, relx=0.67, relwidth=0.05, relheight=0.05)
inp13 = Entry(root)
inp13.place(rely=0.39, relx=0.67, relwidth=0.05, relheight=0.05)
inp14 = Entry(root)
inp14.place(rely=0.47, relx=0.67, relwidth=0.05, relheight=0.05)
inp15 = Entry(root)
inp15.place(rely=0.55, relx=0.67, relwidth=0.05, relheight=0.05)
inp16 = Entry(root)
inp16.place(rely=0.63, relx=0.67, relwidth=0.05, relheight=0.05)
inp17 = Entry(root)
inp17.place(rely=0.71, relx=0.67, relwidth=0.05, relheight=0.05)
inp18 = Entry(root)
inp18.place(rely=0.79, relx=0.67, relwidth=0.05, relheight=0.05)
inp19 = Entry(root)
inp19.place(rely=0.87, relx=0.465, relwidth=0.04, relheight=0.05)
inp20 = Entry(root)
inp20.place(rely=0.87, relx=0.68, relwidth=0.04, relheight=0.05)

#input label
lb1 = Label(root, text='ExternalRiskEstimate',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb1.place(rely=0.15, relx=0.29, relwidth=0.15, relheight=0.05)
lb2 = Label(root, text='MSinceOldestTradeOpen',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb2.place(rely=0.23, relx=0.29, relwidth=0.15, relheight=0.05)
lb3 = Label(root, text='MSinceMostRecentTradeOpen',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb3.place(rely=0.31, relx=0.29, relwidth=0.15, relheight=0.05)
lb4 =Label(root, text='AverageMInFile',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb4.place(rely=0.39, relx=0.29, relwidth=0.15, relheight=0.05)
lb5 = Label(root, text='NumSatisfactoryTrades',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb5.place(rely=0.47, relx=0.29, relwidth=0.15, relheight=0.05)
lb6 = Label(root, text='NumTrades60Ever2DerogPubRec',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb6.place(rely=0.55, relx=0.29, relwidth=0.15, relheight=0.05)
lb7 = Label(root, text='NumTrades90Ever2DerogPubRec',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb7.place(rely=0.63, relx=0.29, relwidth=0.15, relheight=0.05)
lb8 = Label(root, text='PercentTradesNeverDelq',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb8.place(rely=0.71, relx=0.29, relwidth=0.15, relheight=0.05)
lb9 = Label(root, text='MaxDelq2PublicRecLast12M',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb9.place(rely=0.79, relx=0.29, relwidth=0.15, relheight=0.05)
lb10 =Label(root, text='MaxDelqEver',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb10.place(rely=0.15, relx=0.51, relwidth=0.15, relheight=0.05)
lb11 = Label(root, text='NumTotalTrades',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb11.place(rely=0.23, relx=0.51, relwidth=0.15, relheight=0.05)
lb12 = Label(root, text='NumTradesOpeninLast12M',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb12.place(rely=0.31, relx=0.51, relwidth=0.15, relheight=0.05)
lb13 = Label(root, text='PercentInstallTrades',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb13.place(rely=0.39, relx=0.51, relwidth=0.15, relheight=0.05)
lb14 = Label(root, text='NumInqLast6M',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb14.place(rely=0.47, relx=0.51, relwidth=0.15, relheight=0.05)
lb15 = Label(root, text='NumInqLast6Mexcl7days',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb15.place(rely=0.55, relx=0.51, relwidth=0.15, relheight=0.05)
lb16 = Label(root, text='NetFractionRevolvingBurden',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb16.place(rely=0.63, relx=0.51, relwidth=0.15, relheight=0.05)
lb17 = Label(root, text='NumRevolvingTradesWBalance',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb17.place(rely=0.71, relx=0.51, relwidth=0.15, relheight=0.05)
lb18 = Label(root, text='NumInstallTradesWBalance',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb18.place(rely=0.79, relx=0.51, relwidth=0.15, relheight=0.05)
lb19 = Label(root, text='NumBank2NatlTradesWHighUtilization',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb19.place(rely=0.87, relx=0.29, relwidth=0.17, relheight=0.05)
lb20 = Label(root, text='PercentTradesWBalance',padx=3,pady=3,relief="ridge",borderwidth=3,bg='#f5f5f5')
lb20.place(rely=0.87, relx=0.51, relwidth=0.17, relheight=0.05)
root.mainloop()
