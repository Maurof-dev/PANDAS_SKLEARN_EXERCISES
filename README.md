Data file from Tardu,Mehmet and RAHIM,FATIH. (2022). Sirtuin6 Small Molecules. UCI Machine Learning Repository. https://doi.org/10.24432/C56C9Z.



GOAL: find inhibitors (BFEs based) of the Sirtuin6 protein.





The dataset includes 100 molecules;

6 (numerical) descriptors columns as the independent variables;

1 categorical (binary) descriptor, namely high or low BFEs;

#########
#########

The class_model.py script creates a pd object from a datafile and builds a classifier -> decision tree or random forest

The methods include a description of the file (e.g.  column name and type), a histogram plot, a correlation matrix plot, train-test-validation split and 
a function that plots the accuracy score w.r.t. the maximum depth of the classifier.

Other hyperparameter tests will be included.




file description:              SC-5        SP-6        SHBd    minHaaCH     maxwHBa         FMF
count  100.000000  100.000000  100.000000  100.000000  100.000000  100.000000
mean     0.420488    4.429250    0.356504    0.443628    1.919511    0.376232
std      0.195124    1.403348    0.339313    0.138964    0.522992    0.072255
min      0.083333    2.091810    0.000000    0.000000    0.000000    0.153846
25%      0.282118    3.345093    0.000000    0.429809    1.839850    0.326531
50%      0.393350    4.107115    0.373387    0.467655    2.020210    0.376024
75%      0.532954    5.302952    0.482501    0.505934    2.161542    0.423240
max      0.918546    7.641920    1.465000    0.720723    3.778650    0.536585
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   SC-5      100 non-null    float64
 1   SP-6      100 non-null    float64
 2   SHBd      100 non-null    float64
 3   minHaaCH  100 non-null    float64
 4   maxwHBa   100 non-null    float64
 5   FMF       100 non-null    float64
 6   Class     100 non-null    object 
dtypes: float64(6), object(1)
memory usage: 5.6+ KB
data objects: None
variable: SC-5
variable: SP-6
variable: SHBd
variable: minHaaCH
variable: maxwHBa
variable: FMF
variable: Class







