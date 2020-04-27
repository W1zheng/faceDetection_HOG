# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:19:03 2020

@author: WYZ
"""
import numpy as np
from cvxopt import matrix,solvers
from numpy.core.umath import sign
import copy
import math 
import pickle

class svm_model():
    def __init__(self, lagrange=[], sv=[], y=[], b=0, kernel='liner', C=10, load_path=''):
        if (load_path!=''):
            self.load_model(load_path)
        else:
            self.lagrange= copy.deepcopy(lagrange)
            self.sv= copy.deepcopy(sv)
            self.y= copy.deepcopy(y)
            self.b= b
            self.kernel= kernel
            self.C=C
        
    def training(self, X, y):
        "X是N*d维的输入向量"
        "y是输入向量对应的标签"
    
        #数据个数
        N=len(X)
        #系数矩阵
        H=np.zeros((N,N))
        G=np.zeros((2*N,N))
        p=np.zeros((2*N,1))
        q=-1*np.array(np.ones((N,1)))
        A=np.zeros((1,N))
        #二次规划参数
        for i in range(N):
            A[0][i]=y[i]
        for i in range(N):
            for j in range(N):
                if i==j:
                    G[i][j]=-1
                    G[i+N][j]=1
                    p[i+N][0]=self.C
                H[i][j]=y[i]*y[j]*self.Kernel(X[i], X[j], self.kernel)
        #计算二次规划结果
        result=self.cvxopt_solve_qp(H,q,G=G,h=p,A=np.array(A),b=[0.0])
        results=np.array(result['x']).tolist()[:]
        results=[result[0] for result in results]
        lag_max=max(results)
        #print(results)
        for i in range(len(results)):
            if results[i]/lag_max>0.1:
                self.lagrange.append(results[i])
                self.sv.append(X[i])
                self.y.append(y[i])
        #print(self.y)
        t=0
        for i in range(len(self.lagrange)):
            if (self.lagrange[i]<self.C):
                if (self.b==0):
                    self.b=self.y[i]
                    t=i
            self.b-=self.lagrange[i]*self.y[i]*self.Kernel(self.sv[i],self.sv[t],self.kernel)
                        
        
        
    def Kernel(self, x1, x2, kernel, sig= 1):
        "核函数"
        "x1,x2分别是两个样本"
        dot=np.dot(x1,x2)
        if(kernel== 'liner'):
            return dot
        elif(kernel== 'gauss'):
            dif= x1-x2
            return math.exp(-1*sum([x*x for x in dif])/sig)
        
    def Gsvm(self, x):
        "SVM判别函数"
        "x是要预测的向量"
        yhat=self.b
        print(yhat)
        for i in range(len(self.lagrange)):
            yhat+=self.lagrange[i]*self.y[i]*self.Kernel(self.sv[i], x, self.kernel)
        print(yhat)
        return sign(yhat)

    def cvxopt_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
        P = .5 * (P + P.T)  # make sure P is symmetric
        args = [matrix(P), matrix(q)]
        if G is not None:
            args.extend([matrix(G), matrix(h)])
            if A is not None:
                args.extend([matrix(A), matrix(b)])
        sol = solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        return sol
    
    def save_model(self, path):
        f = open(path,'wb')
        data = {'lag':self.lagrange,'sv':self.sv,'y':self.y,
                'b':self.b,'kernel':self.kernel,'C':self.C}
        pickle.dump(data, f)
        
        f.close()
        
    def load_model(self, path):
        f = open(path,'rb')
        para = pickle.load(f)
        self.lagrange = copy.deepcopy(para['lag'])
        self.sv = copy.deepcopy(para['sv'])
        self.y = copy.deepcopy(para['y'])
        self.b = para['b']
        self.kernel = para['kernel']
        self.C = para['C']
        #print(self.lagrange)
        f.close()

# =============================================================================
# China=[[36.0662229200,120.3829882100],
# [37.5134844800,122.1217163900],
# [31.2304324000,121.4737919300],
# [29.9853966500,122.2077802500],
# [27.9949227500,120.6993796200],
# [25.0307240000,121.5200760000],
# [24.4795132400,118.0894773600],
# [22.2780757700,114.1604896100],
# [20.0442268000,110.1998910300],
# [16.8327200000,112.3335600000]]
# 
# Japan=[[42.673348898351485,141.94610595703128],
# [26.14811555530429,127.71469116210937],
# [33.58376949814989,130.55259704589847],
# [33.47727218776036,133.8046869635582],
# [33.853851909674965,130.80191801302135],
# [41.24642297371891,132.16552734375003],
# [34.935526949872816,139.56756591796878],
# [31.817,130.3],
# [32.7502856,129.877667],
# [32.7502856,129.877667]]
# 
# China_inland=[[34.3412614700,108.9398165300],
# [30.5702183700,104.0647735000],
# [30.5927599000,114.3052387800]]
# 
# Japan_inland=[[35.42,139.46],
# [34.4,135.3]]
# 
# ones=np.ones((1,10))                       #生成标注向量y
# minus=-1*np.array(ones)
# y=np.hstack((ones,minus))
# #加入内陆城市标签
# y=np.hstack((y,np.array([[1,1,1,-1,-1]])))
# y=y.tolist()[0]
# X=np.vstack((China,Japan))
# 
# svm=svm(kernel='gauss')
# svm.training(X,y)
# x=[25.7718760000,123.5291260000]
# svm.Gsvm(x)
# =============================================================================
