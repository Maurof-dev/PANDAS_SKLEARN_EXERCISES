import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


import matplotlib
import pylab as plt
import plotly.express as px 
import plotly.graph_objects as go
import seaborn as sns


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'



def data_description(df):
    print('file description:',df.describe())
    print('data objects:',df.info())
    for column in df.columns:
        print('variable:',column)
   



def plot_histo(df):
    for column in df.columns:
        fig = px.histogram(df, 
                        x=column,
                        marginal='box',
                        nbins=47,
                        title=f'Distribution of {column}')

        fig.update_layout(bargap=0.1)
        fig.show()



def plot_corr(df):
    BFE_codes = {'High_BFE': 1, 'Low_BFE': 0}
    df['Class'] = df.Class.map(BFE_codes)
    
    cor = df.corr() 
    sns.heatmap(df.corr(), cmap='Reds', annot=True)
    plt.title('Correlation Matrix');
    plt.show()



    

def splitter(df,testsize,valsize):
    BFE_codes = {'High_BFE': 1, 'Low_BFE': 0}
    df['Class'] = df.Class.map(BFE_codes)
    train_val_df, test_df = train_test_split(df, test_size=testsize, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=valsize, random_state=42)

    input_cols = list(train_df.columns)[0:6]
    target_col = 'Class'

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()
    test_inputs = test_df[input_cols].copy()
    test_targets = test_df[target_col].copy()
    
    X_train = train_inputs
    X_val = val_inputs
    X_test = test_inputs

    return [train_inputs,train_targets,val_inputs,val_targets,test_inputs,test_targets]

# rename some variables for easier handling + build a preliminary model


def random_guess(inputs):
    return np.random.choice([0, 1], len(inputs))




def tester(df,testsize,valsize,depth,modeltype):
    split = splitter(df,testsize,valsize)
    X_train = split[0]
    train_targets = split[1]
    X_val = split[2]
    val_targets = split[3]
    X_test = split[4]
    test_targets = split[5]

    if modeltype == 'decision_tree':
        model = DecisionTreeClassifier(criterion='gini', max_depth=depth,random_state=42)
    if modeltype == 'rnd_forest':
        model = RandomForestClassifier(criterion='gini', max_depth=depth,random_state=42)
    model.fit(X_train, train_targets)

    train_preds = model.predict(X_train)
    pd.value_counts(train_preds)
    train_probs = model.predict_proba(X_train)

    print(modeltype + ' performance:\n')

    print('train score:',model.score(X_train, train_targets))
    print('val. score:',model.score(X_val, val_targets))
    print('random guess score:',accuracy_score(test_targets, random_guess(X_test)))

    if modeltype == 'decision_tree':
        plt.figure(figsize=(10,10))
        plot_tree(model, feature_names=X_train.columns, max_depth=depth, filled=True)
        plt.show()




def depth_check(df,testsize,valsize,modtype):
    split = splitter(df,testsize,valsize)
    X_train = split[0]
    train_targets = split[1]
    X_val = split[2]
    val_targets = split[3]
    X_test = split[4]
    test_targets = split[5]
    depth_list = [i for i in range(1,20)]
    train_score = []
    val_score = []
    test_score = []
    for depth in depth_list:
        if modtype == 'decision_tree':
            model = DecisionTreeClassifier(criterion='gini',max_depth=depth,random_state=42)
        if modtype == 'rnd_forest':
            print('HERE I AM')
            model = RandomForestClassifier(criterion='gini', max_depth=depth,random_state=42)
        model.fit(X_train, train_targets)
        train_score = train_score + [model.score(X_train, train_targets)]
        val_score = val_score + [model.score(X_val, val_targets)]
        test_score = test_score + [model.score(X_test,test_targets)]

    depth_list = np.array(depth_list)
    val_score = np.array(val_score)
    train_score = np.array(train_score)
    test_score = np.array(test_score)
    
    plt.plot(depth_list,val_score,'o-r',label='validation')
    plt.plot(depth_list,train_score,'o-b',label='training')
    plt.plot(depth_list,test_score,'o-g',label='test')
    plt.legend()
    plt.xlabel('depth')
    plt.ylabel('score')
    plt.show()



'''
if __name__ == '__main__':
'''
