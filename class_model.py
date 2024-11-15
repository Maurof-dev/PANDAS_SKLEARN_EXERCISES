import modules_collection
from modules_collection import *








class model_builder():
    def __init__(self, namefile: str = 'SIRTUIN6.csv', testsize: float = 0.25, valsize: float = 0.3, depth: int = 10, modtype: str = 'decision_tree') -> None:
        self.namefile = namefile
        self.testsize = testsize
        self.valsize = valsize
        self.depth = depth
        self.modtype = modtype
    def data_description(self): # column names, column type and statistics
        df = pd.read_csv(self.namefile)
        return modules_collection.data_description(df)
    def plot_histo(self): # plot histogram of all the variables
        df = pd.read_csv(self.namefile)
        return modules_collection.plot_histo(df)
    def plot_corr(self): # correlation matrix
        df = pd.read_csv(self.namefile)
        return modules_collection.plot_corr(df)
    def splitter(self): # training, test and validation split of the dataset
        df = pd.read_csv(self.namefile)
        return modules_collection.splitter(df,self.testsize,self.valsize)
    def tester(self): # accuracy score of a classifier with fixed hyperparameters
        df = pd.read_csv(self.namefile)
        return modules_collection.tester(df,self.testsize,self.valsize,self.depth,self.modtype)
    def depth_check(self): # plot the score vs depth parameter
        df = pd.read_csv(self.namefile)
        return modules_collection.depth_check(df,self.testsize,self.valsize,self.modtype)





modtype = 'decision_tree'
modalt = 'rnd_forest'

obj = model_builder(testsize=0.1,valsize=0.1,modtype=modalt)

obj.data_description()
obj.plot_histo()
obj.plot_corr()
obj.splitter()
obj.tester()
obj.depth_check()

