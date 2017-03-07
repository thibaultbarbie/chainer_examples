# Author : Thibault Barbie

import numpy as np
import chainer
from chainer import Variable, optimizers, serializers,iterators,reporter
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt

n_epoch = 10
batchsize = 100   
n_repetition=3

class NN(chainer.Chain):
    def __init__(self, hid_n):
        super(NN, self).__init__(
            lstm=L.LSTM(1,hid_n),
            lin=L.Linear(hid_n,2),
        )

    def reset_state(self):
        self.lstm.reset_state()
        
    def __call__(self,x):
        one_real_to_vector=L.Linear(1,100,initialW=np.ones((100,1)))
        W=np.concatenate((np.ones((100,1)),np.zeros((100,1))),axis=1)
        two_reals_to_vector=L.Linear(2,100,initialW=W)
        
        t=Variable((np.ones((len(x),100))*np.linspace(0,1,num=100)).astype(np.float32))
        t1=Variable((-np.ones((len(x),100))*0.4).astype(np.float32))
        function_x=0.4*F.exp(-90*pow(t-one_real_to_vector(x),2))+F.exp(-90*pow(t-t1-one_real_to_vector(x),2))
        
        function_h=0
        h=x
        for i in range(n_repetition):
            l=L.Linear(1,100,initialW=np.ones((100,1)))
            
            h=self.lstm(x)
            h=F.sigmoid(self.lin(h))
            
            function_h+=l(h[:,1:2])*F.exp(-90*pow(t-two_reals_to_vector(h),2))
            
            
        loss=F.mean_squared_error(function_h,function_x)
        reporter.report({'loss': loss}, self)
        self.reset_state()
        return loss

    def forward(self,x):
        h=x
        result=[]
        for i in range(n_repetition):
            h=self.lstm(x)
            h=F.sigmoid(self.lin(h))
            result.append(h.data)
        return result

model=NN(50)
optimizer = optimizers.Adam(alpha=0.001)
optimizer.setup(model)

Xtr=np.random.rand(50000,1).astype(np.float32)
train = zip(Xtr,Xtr)
train=Xtr
train_iter = iterators.SerialIterator(train, batch_size=batchsize, shuffle=True)
updater = chainer.training.StandardUpdater(train_iter, optimizer)
trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.run()

def f_real(t,x):
    result=0.4*np.exp(-90*pow(t-x[0],2))+np.exp(-90*pow(t-x[0]+np.ones(100)*0.4,2))
    return result


def gaussian(t,x):
    result=x[0,1]*np.exp(-90*pow(t-x[0,0],2))
    return result

x=np.asarray([[0.64]],dtype=np.float32)
t=np.linspace(0,1,num=100)
y1=f_real(t,x)

approx_param=model.forward(x[0:1])
print approx_param
y2=0
for param in approx_param:
    y2+=gaussian(t,param)
plt.plot(t,y1,label='real function')
plt.plot(t,y2,label='approximation')
plt.legend()
plt.ylim([0,1.3])
plt.savefig('result.png')


