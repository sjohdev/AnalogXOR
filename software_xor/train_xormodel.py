import torch
import torch.optim as optim
from torch import nn
import xorModel as xm
import xor_helpers as xorh

# Pytorch tensor dims = (batchsz, numchannels, height, width)
device = "cpu"

if __name__ == '__main__': 
    # Get training data
    training_tuple = xorh.get_trainingdata()
    X = training_tuple[0]
    y = training_tuple[1]

    # Define test data (same as training data in this particular case)
    X_test = X
    y_test = y

    # define batch size (each batch contains all datapoints in this case)
    batch_size = len(y)

    # get model:
    model = xm.xorModel().to(device)

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
    pred_labels = xorh.test_model(model, X_test, y_test)

    # Print model weights for each layer
    xorh.print_weights(model)
        
    # Save model  
    #torch.save(model.state_dict(), "xormodel2.pth")        
