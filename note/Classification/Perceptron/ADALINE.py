import numpy as np

class AdalineGD():
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
        
    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.cost_=[]
        
        for i in range(self.n_iter):
            net_input=self.net_input(X)
            # 決定関数
            output=self.activation(net_input)
            # 誤差の計算
            errors=(y-output)
            # 重みの更新
            # 特徴量ベクトルXと誤差ベクトルerrorの積
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            # コストの計算
            cost =(errors**2).sum()/2.0
            self.cost_.append(cost)
            
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    
    def activation(self,X):
        return X