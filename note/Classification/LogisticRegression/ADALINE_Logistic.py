import numpy as np
class LogisticRegressionGD():
    def __init__(self,eta=0.05,n_iter=100,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.randome_state=random_state
        
    def fit(self,X,y):
        rgen=np.random.RandomState(self.randome_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            net_input=self.net_input(X)
            output=self.activation(net_input)
            errors=(y-output)
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=-y.dot(np.log(output))-((1-y).dot(np.log(1-output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    
    def activation(self,z):
        return 1./(1.+np.exp(-np.clip(z,-250,250)))
    
    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,0)