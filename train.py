from torch.optim import Adam
from tqdm import tqdm
from torch import nn
import torch 
from src.toxic_bert import Dataset, BertClassifier
import argparse
from src.preprocessing import preprocess
import pandas as pd


def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    
                    output = model(input_id, mask)
                    
                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataloader.dataset): .3f} \
                | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
            

def main(df, data_name='text', label_name='class_true', epochs=2, lr=2e-5, batch_size=8):
    
    df = preprocess(df, data_name=data_name, label_name=label_name)

    df_train = df.sample(frac=0.85, random_state=25)
    df_val = df.drop(df_train.index)

    print(f"No. of training examples: {df_train.shape[0]}")
    print(f"No. of testing examples: {df_val.shape[0]}")

    train_ds, val_ds = Dataset(df_train, label_name, data_name), Dataset(df_val, label_name, data_name)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    model = BertClassifier()
    
    torch.cuda.empty_cache()

    train(model, train_dl, val_dl, lr, epochs)

    torch.save(model, 'model.pt')
    torch.save(model.state_dict(), 'model_dict.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--data_name', default='text')
    parser.add_argument('--label_name', default='class_true')
    parser.add_argument('--epochs', default=2)
    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--batch_size', default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    main(df, data_name=args.data_name, label_name=args.label_name, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)