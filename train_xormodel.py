import torch
import torch.optim as optim
from torch import nn
import numpy as np
import pickle
import xorModel

# Pytorch tensor dims = (batchsz, numchannels, height, width)
device = "cpu"

if __name__ == '__main__': 
    # Define training data
    # input: X
    X = torch.zeros(4, 2, dtype=torch.float32)
    X[1,1] = 1
    X[2,0] = 1
    X[3,0] = 1
    X[3,1] = 1
    print('training input:')
    print(X)

    # output/labels: y
    y = torch.zeros(4, dtype=torch.long)
    y[1] = 1
    y[2] = 1
    print('training labels:')
    print(y)

    # Define test data (same as training data in this particular case)
    X_test = X
    y_test = y

    # define batch size (each batch contains all datapoints in this case)
    batch_size = len(y)

    # get model:
    model = xorModel.xorModel().to(device)

    # train model
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)

    model.train()
    maxiter = 1000
    for epoch in range(0, maxiter):
        # forward + backward + optimize
        outputs = model(X)
        #print(outputs)
        loss = loss_fcn(outputs, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        # zero the parameter gradients
        optimizer.zero_grad()

        # print statistics
        print(f'loss: {loss}')

    print('Finished training')

    # Compare model classifications against labels
    model.eval()
    with torch.no_grad():
        logits_out = model(X_test)
    pred_labels = logits_out.argmax(1)
    print(f'Input data:\n{X_test}')
    print(f'Predicted labels:\n{pred_labels}')
    print(f'True labels:\n{y_test}')

    # Print model weights for each layer
    print('\nPrinting weights and bias for each layer: ')
    layers = model.xor_stack
    for i in range(0, len(layers)):
        if 'weight' in dir(layers[i]):
            print(f'------------------\nLayer {i}:\nWeights:\n{layers[i].weight}')
        if 'bias' in dir(layers[i]):
            print(f'Bias:\n{layers[i].bias}')

        
    # Save model, method 1   
    #torch.save(model.state_dict(), "xormodel2.pth")        

    # Save model, method 2
    #with open('xorNetwork.pkl', 'wb') as f2:
    #        pickle.dump(model, f2)
