import sys
import argparse
from src.toxic_bert import BertClassifier, Dataset
from src.preprocessing import preprocess
import torch
import pandas as pd

def evaluate(model, test_data, data_name='text', label_name='class_true'):
    preds = list()

    test = Dataset(test_data, label_name=label_name, data_name=data_name)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    model.eval()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)
              #print(output)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
              
              #preds.append(output.argmax(dim=1))

              for x in output.argmax(dim=1):
                  preds.append(x.item())

              #print(output.argmax(dim=1).item())
              #print(output)
              #print(preds)
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    return preds


def inference(df, data_name='text', label_name='class_true', model=None, model_path='model_dict.pt'):
    if model is None:
        model = BertClassifier()
        use_cuda = torch.cuda.is_available()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if use_cuda else "cpu") ))
        model.eval()

    df = preprocess(df, data_name=data_name, label_name=label_name)
    preds = evaluate(model, df, data_name, label_name)
    df.assign(class_prediction=preds)
    print(df.head(20))
    pd.to_csv('output.csv')



df = pd.read_csv('data_test_public.csv')
inference(df, data_name='comment', label_name='toxic')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--model_path')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    inference(df, data_name='comment', label_name='toxic')

    
    