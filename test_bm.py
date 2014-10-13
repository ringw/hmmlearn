from hmmlearn import hmm
from pylab import *
from numpy import random

fwd = np.eye(5).astype(bool)
rev = fwd[::-1]

seqs = []
for i in xrange(250):
    A = []
    B = []
    for j in xrange(20):
        start = random.randint(0, 1)
        end = random.randint(3, 5)
        label = random.randint(0, 2)
        whichone = (fwd, rev)[label]
        arr = whichone[start:end].copy()
        arr ^= random.random(arr.shape) < 0.02
        A.append(arr)
        B += [label for i in xrange(end-start)]
    seqs.append((concatenate(A), B))

h = hmm.BernoulliMultiHMM(50)
h.fit(seqs[1:])
a, b = seqs[0]
n = -np.ones(len(b), int)
figure()
plot(np.exp(h._log_multinomprob[h.predict((a, n)), -1]), 'r')
plot(b, 'g')

C = [np.hstack([a,np.array(b,bool)[:,None]]) for a,b in seqs]

if 0:
    h2 = hmm.BernoulliHMM(20)
    h2.fit(C)
    figure()
    plot(np.exp(h2._log_emissionprob[h2.predict(C[0]), -1]), 'r')
    C[0][:,-1] = 0
    plot(np.exp(h2._log_emissionprob[h2.predict(C[0]), -1]), 'y')
    C[0][:,-1] = 1
    plot(np.exp(h2._log_emissionprob[h2.predict(C[0]), -1]), 'b')
    plot(C[0][:,-1], 'g-')

show()
