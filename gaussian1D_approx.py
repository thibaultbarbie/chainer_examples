# Author : Thibault Barbie

import numpy as np
import chainer
from chainer import Variable, optimizers, serializers,iterators,reporter
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt

n_epoch = 10
batchsize = 10   

class NN(chainer.Chain):
    def __init__(self, input_n, output_n):
        super(NN, self).__init__(
            l1=L.Linear(input_n, output_n),
        )

    def __call__(self,x,y):
        h=self.l1(x)
        
        real_to_vector=L.Linear(1,100,initialW=np.ones((100,1)))
        t=Variable((np.ones((len(x),100))*np.linspace(0,1,num=100)).astype(np.float32))

        function_x=F.exp(-90*pow(t-real_to_vector(x),2))
        function_h=F.exp(-90*pow(t-real_to_vector(h),2))

        loss=F.mean_absolute_error(function_h,function_x)
        reporter.report({'loss': loss}, self)
        
        return loss

    def forward(self,x):
	h = self.l1(x)
        return h

model=NN(1,1)
optimizer = optimizers.Adam(alpha=0.001)
optimizer.setup(model)

Xtr=np.random.rand(5000,1).astype(np.float32)
train = zip(Xtr,Xtr)

train_iter = iterators.SerialIterator(train, batch_size=batchsize, shuffle=True)
updater = chainer.training.StandardUpdater(train_iter, optimizer)
trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.run()

def f(alpha,x):
    result=np.exp(-90*pow(alpha-x[0],2))
    return result

x=np.asarray([[0.64]],dtype=np.float32)
t=np.linspace(0,1,num=74)
y1=f(t,x)

y2=f(t,model.forward(x[0:1]).data)
plt.plot(t,y1,label='real function')
plt.plot(t,y2,label='approximation')
plt.legend()
plt.ylim([0,1.3])
plt.savefig('result.png')
