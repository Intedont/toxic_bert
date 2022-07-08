import sys
import argparse
from src.toxic_bert import BertClassifier, Dataset
from src.preprocessing import preprocess
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

def evaluate(model, test_data, batch_size=16, data_name='text', label_name='class_true'):
    '''Непосредственно функция которая выполняет вычисления
    Принимает на вход
    - pytorch модель
    - предобработанные данные в виде датафрейма
    - размер батча
    - имя столбца с данными
    - имя столбца с ответами
    Возвращает два списка - список с предсказанными ответами и списов с вероятностями
    '''
    preds = list()
    probs = list()

    test = Dataset(test_data, label_name=label_name, data_name=data_name)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    model.eval()

    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader):
            
            test_label = test_label.to(device)
              
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
              
            for x in output:
                preds.append(x.argmax().item())
                probs.append(F.softmax(x, dim=-1).max().item())

    return preds, probs


def inference(data_path, data_name='text', label_name='class_true', model=None, model_path='model_dict.pt'):
    '''Главная функция модуля - позволяет вычислить предсказания для переданного csv файла
    Помимо пути к csv файлу, принимает на вход: 
    - название столбца с данными data_name
    - название столбца с правильными классами label_name
    - pytorch модель
    или
    - путь к сохраненным весам модели

    '''
    
    source_df = pd.read_csv(data_path)

    if model is None:
        model = BertClassifier()
        use_cuda = torch.cuda.is_available()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if use_cuda else "cpu") ))
        model.eval()

    df = source_df.copy()
    df = preprocess(df, data_name=data_name, label_name=label_name)
    preds, probs = evaluate(model, df, data_name=data_name, label_name=label_name)

    source_df = source_df.assign(class_prediction=preds)
    source_df = source_df.assign(probabilities=probs)
    
    return source_df


#пример вызова inference()
#df = pd.read_csv('data_test_public.csv')
#inference(df, data_name='comment', label_name='toxic')


if __name__ == '__main__':
    '''Позволяет запускать скрипт из консоли 
    принимает параметры 
    --data_path - путь к csv файлу с данными
    --model_path - путь к файлу с весами модели
    --data_name - имя столбца с данными
    --label_name - имя столбца с ответами
    Пример скрипта
    python -m predict_csv --data_path data_test_public.csv --data_name comment --label_name toxic 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--model_path', default='model_dict.pt')
    parser.add_argument('--data_name', default='text')
    parser.add_argument('--label_name', default='class_true')
    args = parser.parse_args()

    ans = inference(args.data_path, data_name=args.data_name, label_name=args.label_name, model_path=args.model_path)
    print(ans.head(10))
    ans.to_csv('output.csv')
