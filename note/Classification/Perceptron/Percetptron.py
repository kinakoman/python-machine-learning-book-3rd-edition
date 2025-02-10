import numpy as np

class Perceptron():
    def __init__(self,eta=0.01,n_inter=50,random_state=1):
        # 学習率
        self.eta=eta
        # 学習回数
        self.n_inter=n_inter
        # 乱数シード
        self.random_state=random_state
    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state)
        # 重みの初期化
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.errors_=[]
        
        for _ in range(self.n_inter):
            errors=0
            # 学習データの数だけ重み更新を行う
            for xi,target in zip(X,y):
                # 重み更新の計算
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
                
            self.errors_.append(errors)
        return self
    def net_input(self,X):
        # 総入力を計算
        # 重みと訓練データの積にbiasを足す
        return np.dot(X,self.w_[1:])+self.w_[0]
    def predict(self,X):
        # 決定関数によるクラスラベルの付与
        return np.where(self.net_input(X)>=0.0,1,-1)


        