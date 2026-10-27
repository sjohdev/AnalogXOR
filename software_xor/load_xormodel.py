import xor_helpers as xorh
import xorModel as xm
import torch

# Pytorch tensor dims = (batchsz, numchannels, height, width)

if __name__ == '__main__': 
    # Get training data
    training_tuple = xorh.get_trainingdata()
    X = training_tuple[0]
    y = training_tuple[1]

    # Define test data (same as training data in this particular case)
    X_test = X
    y_test = y

    # get model:
    device = "cpu"
    model = xm.xorModel().to(device)
    model.load_state_dict(torch.load('xormodel2.pth', weights_only=True))

    # Compare model classifications against labels
    pred_labels = xorh.test_model(model, X_test, y_test)

    # Print model weights for each layer
    xorh.print_weights(model)
