import pandas as pd
import torch
import argparse
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from sound_classification_dataset import SoundDS
from sound_classification_model import AudioClassifier

# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
def prepara_data():
    data_path = 'UrbanSound8k'

    # Read metadata file
    metadata_file = data_path + '/UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Construct file path by concatenating fold and file name
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    myds = SoundDS(df, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)

    return (train_dl, val_dl)


# ----------------------------
# Training Loop
# ----------------------------
def training(train_dl, num_epochs):
    # Tensorboard 
    writer = SummaryWriter()

    # Create the model and put it on the GPU if available
    model = nn.DataParallel(AudioClassifier())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            #if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        avg_acc = correct_prediction/total_prediction
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Acc/train", avg_acc, epoch)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}')
    
    torch.save(model.state_dict(), 'model.pt')    
    print('Finished Training')

# ----------------------------
# Inference
# ----------------------------
def inference (model, test_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True)
    args = vars(ap.parse_args)

    train_dl, test_dl = prepara_data()
    if args['mode'] == 'train':
        # Run training model
        training(train_dl, num_epochs=100)
    else:
        # Run inference on trained model with the validation set load best model weights
        # Load trained/saved model
        model_inf = nn.DataParallel(AudioClassifier())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_inf = model_inf.to(device)
        model_inf.load_state_dict(torch.load('model.pt'))
        model_inf.eval()

        # Perform inference
        inference(model_inf, test_dl)

