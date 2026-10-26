import torch

# Helper functions for the XOR neural-network

def test_model(model, X_test, y_test):
    # Compare model classifications against labels
    model.eval()
    with torch.no_grad():
        logits_out = model(X_test)
    pred_labels = logits_out.argmax(1)
    print(f'Input data:\n{X_test}')
    print(f'Predicted labels:\n{pred_labels}')
    print(f'True labels:\n{y_test}')
    return(pred_labels)

def print_weights(model):
    # Print model weights for each layer
    print('\nPrinting weights and bias for each layer: ')
    layers = model.xor_stack
    for i in range(0, len(layers)):
        if 'weight' in dir(layers[i]):
            print(f'------------------\nLayer {i}:\nWeights:\n{layers[i].weight}')
        if 'bias' in dir(layers[i]):
            print(f'Bias:\n{layers[i].bias}')

def get_trainingdata():
    # Create training data
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
    return((X, y))
