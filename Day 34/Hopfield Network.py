import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class HopefieldNetwork(object):
    def __init__(self, size):
        self.num_neurons = size
        self.W = np.ones((self.num_neurons, self.num_neurons), dtype = 'float16')
    
    def resetNetwork(self):
        self.W = np.ones((self.num_neurons, self.num_neurons), dtype = 'float16')
    
    def trainHebb(self, data, showWeights = False):
        self.type = self.trainHebb

        if showWeights:
            self.initialize_for_showing_weights()

        X = np.array(data).T
        self.W = (np.matmul(X,X.T) - len(data)*np.eye(self.num_neurons, dtype='float16'))/self.num_neurons

        if showWeights:
            self.show_weights(f'Final', block=True)
            self.save_weights()

    def trainOja(self, data, u = 0.0001, iter = 1000, showWeights = False):
        self.type = self.trainOja

        self.trainHebb(data)

        if showWeights:
            self.initialize_for_showing_weights()
            self.show_weights(f'Iteration#{0}')

        for it in range(iter):
            Wprev = self.W.copy()
            for x in data:
                x = (x+1)/2
                y = np.matmul(x, self.W)
                for i in range(self.num_neurons):
                    for j in range(self.num_neurons):
                        self.W[i,j] += u*(y[i]*x[j] - self.W[i,j]*(y[i]**2))

            if(showWeights):
                self.show_weights(f'Iteration#{it+1}')
                print(f'Iteration #{it+1}/{iter}', "\t", np.linalg.norm(Wprev - self.W))

            if np.linalg.norm(Wprev - self.W) < 1e-14:
                break 
        
        self.W -= np.diag(np.diag(self.W))
        if showWeights:
            self.show_weights(f'Final', block=True)
            self.save_weights()

    def _async(self, x, W):
        xsync, _ = self._sync(x, W)
        change = np.multiply(xsync, x)
        changed = np.argwhere(change < 0)
        if changed.size ==0:
            return np.array(x), False
        else:
            toChange = np.random.choice(changed.reshape(-1))
            x[toChange] = xsync[toChange]
            return np.array(x), True 

    def _sync(self, x, W):
        return np.sign(np.matmul(W, x)), True

    def forward(self, data, iter = 20, asyn = False, print = None):
        s = data
        if asyn:
            f = self._async
        else:
            f = self._sync
        
        e = []
        e.append(0)
        e.append(self.energy(s))

        for i in range(iter):
            if self.type == self.trainOja:
                s = (s+1)/2
            s, cont = f(s, self.W)
            e.append(self.energy(s))

            if print != None:
                # print(s, f'{i+1}')
                print(s, f'{i+1}  E: {e[-1]}')
                self.plotEnergy(e)

            # if not cont:
            #     return s

            # if f == self._sync and e[-1] == e[-2]:
            #     return s
        
        return s

    def energy(self, s):
        return -0.5*np.matmul(np.matmul(s, self.W), s)

    def plotEnergy(self, energy):
        plt.figure(5000)
        # print(energy[-3], energy[-2], energy[-1])
        plt.ion()
        plt.clf()
        plt.title('Energy value')
        plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.plot(energy)

    def initialize_for_showing_weights(self):
        plt.ion()
        plt.figure(figsize=(6, 5))
        plt.tight_layout()

    def show_weights(self, iteration_name, block = False):
        plt.figure(1)
        plt.clf()
        plt.title(f'Network Weights - {iteration_name}')
        colors = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(colors)

        plt.show(block = block)
        plt.pause(0.1)

    def save_weights(self):
        plt.figure(1)
        plt.clf()
        plt.title(f'Network Weights - Final')
        colors = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(colors)
        plt.savefig('weights.png')