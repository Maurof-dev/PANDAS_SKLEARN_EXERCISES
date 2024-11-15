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












