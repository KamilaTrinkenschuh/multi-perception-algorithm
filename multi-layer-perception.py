import numpy as np
class MLP(object):
    def func_id(self,x):
        return x
    def func_sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    def func_relu(self,x):
        return np.maximum(x,0)
    def __init__(self, n_input_neurons = 2, n_hidden_neurons = 2, n_output_neurons = 1, weights = None, *args, **kwargs):
        self.n_input_neurons=n_input_neurons
        self.n_hidden_neurons=n_hidden_neurons
        self.n_output_neurons=n_output_neurons
        self.weights=weights
        W_IH=[]
        W_HO=[]
        self.network=[]
        self.inputLayer  = np.zeros((self.n_input_neurons+1,1))
        self.inputLayer[0]=1.0
        self.network.append(self.inputLayer)
        if weights:
            W_IH=self.weights[0]
        else: 
            W_IH=np.zeros((self.n_hidden_neurons+1, self.n_input_neurons+1))
        self.network.append(W_IH)
        self.hiddenLayer  = np.zeros((self.n_input_neurons+1,3))
        self.hiddenLayer[0]=1.0
        self.network.append(self.hiddenLayer)
        if weights:
            W_HO=self.weights[1]
        else: 
            W_HO=np.zeros((self.n_output_neurons+1,self.n_hidden_neurons+1))
        self.network.append(W_HO)
        self.output_Layer=np.zeros((self.n_output_neurons+1,3))
        self.output_Layer[0]=0.0
        self.network.append(self.output_Layer)
    def print(self):
        print("Multi-Layer-Perception")
        np.set_printoptions(\
            formatter={'float': lambda x: "{0:0.3f}".format(x)})
        for idx, nn_part in enumerate(self.network):
            print(nn_part)
            print('___________________><_____________________')
    def predict(self,x):
    
        self.network[0][:,0]=x
        self.network[2][1:,0]=np.dot(self.network[1][1:,:], self.network[0][:,0])
        self.network[2][1:,1]=self.func_sigmoid(self,network[2][1:,0])
        self.network[2][1:,2]=self.func_id(self.network[2][1:,1])
            self.network[4][1:,0]=np.dot(self.network[3][1:,:], self.network[2][:,2])
            self.network[4][1:,1]=self.func_sigmoid(self,network[4][1:,0])
            self.network[4][1:,2]=np.round(self.func_id(self.network[4][1:,1]))
        return self.network[4][1:,2]

W_IH = np.matrix([[0.0,0.0,0.0],[-10,20.0,20.0],[30,-20.0,-20.0]])
W_HO = np.matrix([[0.0,0.0,0.0],[-30,20.0,20.0]])
weights=[]
weights.append(W_IH)
weights.append(W_HO)
nn= MLP(weights=weights)
nn.print()
X=np.array([[1.0,1.0,1.0],[1.0,0,1.0],[1.0,1.0,0],[1.0,0,0]])
y=np.array([0,1.0,1.0,0])
print('Predict:')
for idx,x in enumerate(X):
    print('{} {} -> {}'.format(x,y[idx],nn.predict(x)))
