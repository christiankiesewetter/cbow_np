import numpy as np
import re

def tokenize(sequences, thresh = 1, unk_token = '--unk--'):
    vocab = set([unk_token])
    vocab_frequencies = {unk_token:0}
    for sequence in sequences:
        for word in re.findall(r'\w+',sequence):
            if not word in vocab:
                vocab.add(word)
            vocab_frequencies[word] = vocab_frequencies.get(word, 0) +1

    remove_v = []
    if thresh != None:
        for word in vocab:
            if word != unk_token and vocab_frequencies[word] < thresh:
                remove_v.append(word)
    for word in remove_v:
        vocab.remove(word)

    word2id = {word:wid for wid, word in enumerate(vocab)}
    tokenized = []
    for sequence in sequences:
        tok_seq = []
        for word in re.findall(r'\w+',sequence):
            tok_seq.append(word2id[word] if word in vocab else word2id[unk_token])
        tokenized.append(tok_seq)

    return tokenized, word2id


def create_cbow_examples(sequences, k=1, examples = []):
    for sequence in sequences:
        for ii in range(k, len(sequence)-k):
            example = (sequence[ii-k:ii] + sequence[ii+1:ii+k+1], [sequence[ii]])
            examples.append(example)
    return list(zip(*examples))




class CBOWNet:

    def init_hidden_layer(self):
        w = np.random.random((self.V, self.hidden_dims))
        b = np.random.random((1, self.hidden_dims))
        self.hidden_layer['l0'] = w, b, self.relu

    def init_output_layer(self):
        w = np.random.random((self.hidden_dims, self.V))
        b = np.random.random((1, self.V))
        self.hidden_layer['l1'] = w, b, self.softmax

    def one_hot_encode(self, X):
        return self.eye[X].squeeze()

    def relu(self, X, cache = None, fw = True):
        if fw : 
            return np.maximum(X, 0)
        else: 
            X[cache <= 0] = 0
            return X

    def cross_entropy_loss(self, y, y_hat):
        return - (y * np.log(y_hat)).sum(axis=1)

    def softmax(self, X, cache = None, fw = True):
        if fw:
            X_exp = np.exp(X)
            return X_exp / X_exp.sum(axis=1, keepdims=True)
        else:
            return X * (1 - X)

    def grad_desc(self, dA, cache, m, lid):
        X, Z = cache[f'l{str(lid-1)}']
        # activation cache
        w, b, activation = self.hidden_layer[f'l{lid}']
        
        dz = activation(dA, cache=X, fw=False)
        # linear
        dW = np.dot(dz.T, Z) / m
        db = dz.sum(axis=0, keepdims=True) / m
        dA = np.dot(w, dz.T)
        return dW.T, db, dA.T

    def fw_layer(self, xin, lid):
        w, b, activation = self.hidden_layer[lid]
        X_h = np.dot(xin, w) + b
        Z_h = activation(X_h, fw=True)
        return X_h, Z_h


    def forward(self, X, train = False):
        if train:
            cache = {}
            # first layer
            cache['l0'] = self.fw_layer(X, 'l0')
            # second layer
            cache['l1'] = self.fw_layer(cache['l0'][1], 'l1')
            return cache

        else:
            _, X = self.fw_layer(X, 'l0')
            _, X = self.fw_layer(X, 'l1')
            return X
    

    def backward(self, x, y, cache, lrate):
        m = y.shape[0]
        dW, db, dA = None, None, cache['l1'][1] - y
        for lid in reversed(range(1, len(self.hidden_layer))):
            dW, db, dA = self.grad_desc(dA, cache, m, lid)

            w, b, activation = self.hidden_layer[f'l{str(lid)}']
            w = w - lrate * dW
            b = b - lrate * db
            self.hidden_layer[f'l{str(lid)}'] = w, b, activation



    def train(self, data, epochs=2, batch_size=10):
        x, y = data
        m = y.shape[0]
        lrate = 1.0 * self.alpha
        for epoch in range(epochs):
            lrate = self.alpha * (1-epoch/epochs)
            eploss = 0.0
            for ii in range(m//batch_size):
                batchstep = slice(ii, ii + batch_size)
                x_0, y_0  = x[batchstep],y[batchstep]
                # preprocess x,y
                x_ , y_ = self.one_hot_encode(x_0), self.one_hot_encode(y_0)
                x_ = x_.mean(axis=1, keepdims=True).squeeze()
                cache = self.forward(x_, train=True)
                
                y_hat = cache['l1'][1]
                J = self.cross_entropy_loss(y_, y_hat)
                
                eploss += J.mean()
                self.backward(x_, y_, cache, lrate)

            print(f'Epoch {epoch+1}: {eploss:.5f}, lr: {lrate:.4f}', end='\r')
            eploss = 0.0


    def __init__(self, V, embed_dims, hidden_dims, alpha = 1e-3):
        self.alpha = alpha
        self.isbuilt = False
        
        self.V = V
        self.hidden_layer = {}

        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.eye = np.eye(V)
        
        self.init_hidden_layer()
        self.init_output_layer()


    def __call__(self, X):
        return self.forward(X)


        


if __name__ == '__main__':

    phrases = [ 'This is funny',
                'This is great, I like it, got it',
                'there is nothing better, you got to like it',
                'it\'s better like that']

    tokenized, vocab_count_id = tokenize(phrases)
    cbowx, cbowy = create_cbow_examples(tokenized, k=1)
    cbowx = np.array(cbowx)
    cbowy = np.array(cbowy)

    V = len(vocab_count_id)
    m = 5
    hidden_layer = 10
    net = CBOWNet(V, m, hidden_dims=10, alpha=1e-02)
    net.train((cbowx, cbowy), epochs = 3000, batch_size=5)
