import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HiGT")
    parser.add_argument("--all_data_path", type=str, default="Path of Graph Data")
    parser.add_argument("--label_path", type=str, default="Path of label File")
    parser.add_argument("--one_layer", type=bool, default=True)
    parser.add_argument("--repeat_num", type=int, default=1, help="Number of repetitions of the experiment")
    parser.add_argument("--divide_seed", type=int, default=2023, help="Seed")
    parser.add_argument("--drop_out_ratio", type=float, default=0.2, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate of model training")
    parser.add_argument("--epochs", type=int, default=50, help="Cycle times of model training")
    parser.add_argument("--batch_size", type=int, default=8, help="Data volume of model training once")
    parser.add_argument("--saved_model_path", type=str, default="your save path",
                        help="Save the path prefix of the model")
    parser.add_argument("--out_classes", type=int, default=256, help="Model middle dimension")
    parser.add_argument("--pool_ratio_list", type=list, default=[0.5, 5], help="Proportion of the first pool")
    parser.add_argument("--gcn_channel_list", type=list, default=[1024, 1024, 1024], help="number of channels in gcn")
    # mhit_num
    parser.add_argument("--mhit_num", type=int, default=3, help="number of HIViT block")
    parser.add_argument("--fusion_exp_ratio", type=int, default=4, help="expansion ratio of fusion block")
    parser.add_argument("--mpool_method", type=str, default="global_mean_pool", help="Global pool method")
    parser.add_argument("--log", type=str, default="No log", help="log of this experiment")
    parser.add_argument("--fold_num", type=int, default=5, help="fold number of this experiment")
    parser.add_argument('--num_classes', default=2, type=int, help='number of output classes')

    args, _ = parser.parse_known_args()
    return args

import sys
import time
import torch
import joblib
import random
import os.path
import argparse
import numpy as np
import time as sys_time
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix, recall_score, f1_score, \
    classification_report
from HiGT import HiGT
# import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def return_train_val_index(train_index, label_list, seed):
    label_list_temp = []
    for index in train_index:
        label_list_temp.append(label_list[index])
    train_index_split, val_index_spilt, _, _ = train_test_split(train_index, label_list_temp, test_size=0.25,
                                                                random_state=seed, stratify=label_list_temp)
    return train_index_split, val_index_spilt


def reuturn_data_label(all_data, data_index, patiens_list, label_list, batch_size):
    data_array = []
    label_array = []
    for item in data_index:
        temp_patient_name = patiens_list[item]
        temp_data = all_data[temp_patient_name]
        label_array.append(label_list[item])
        data_array.append(temp_data)
    step = batch_size
    data_array_temp = [data_array[i:i + step] for i in range(0, len(data_array), step)]
    label_array_temp = [label_array[i:i + step] for i in range(0, len(label_array), step)]
    return data_array_temp, label_array_temp


def return_acc(prediction_array, label_array):
    prediction_array = np.array(prediction_array)
    label_array = np.array(label_array)
    # print(prediction_array, label_array)
    correct_num = (prediction_array == label_array).sum()
    len_array = len(prediction_array)
    return correct_num / len_array


def return_auc(possibility_array, label_array, args):
    np_label_array = np.zeros((len(label_array), args.num_classes))
    for i in range(len(label_array)):
        np_label_array[i][label_array[i]] = 1
    possibility_array = np.array(possibility_array)

    aucs = []
    for c in range(0, args.num_classes):
        label = np_label_array[:, c]
        prediction = possibility_array[:, c]
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)

    total_auc_macro = roc_auc_score(np_label_array, possibility_array, average="macro")
    total_auc_micro = roc_auc_score(np_label_array, possibility_array, average="micro")
    return aucs, total_auc_macro, total_auc_micro

def val_test_block(model, loss_fun, device, data_for_val_test, label_for_val_test, args):
    model.eval()
    with torch.no_grad():
        label_array_for_val_test = []
        prediction_array_for_val_test = []
        possibility_array_for_val_test = []
        total_loss_for_val_test = 0
        for index_batch_for_val_test, data_batch_for_val_test in enumerate(data_for_val_test):
            label_batch_for_val_test = label_for_val_test[index_batch_for_val_test]
            for index_temp, data_val_test_single in enumerate(data_batch_for_val_test):
                label_val_test_single = label_batch_for_val_test[index_temp]
                label_array_for_val_test.append(label_val_test_single)
                data_val_test_single = data_val_test_single.to(device)
                label_val_test_single = torch.tensor([label_val_test_single]).to(device)
                
                Y_prob = model(data_val_test_single)

                output_for_val_test = Y_prob
                prediction_for_val_test = torch.argmax(Y_prob)
                loss_for_val_test = loss_fun(output_for_val_test,F.one_hot(label_val_test_single, num_classes=args.num_classes).squeeze().float())
                
                total_loss_for_val_test += loss_for_val_test
                prediction_array_for_val_test.append(prediction_for_val_test.cpu().item())
                possibility_array_for_val_test.append(output_for_val_test.squeeze().cpu().detach().numpy())
        acc_for_val_test = return_acc(prediction_array_for_val_test, label_array_for_val_test)
        auc_for_val_test, macro_auc_val_test, micro_auc_val_test = return_auc(possibility_array_for_val_test, label_array_for_val_test, args)

        # return label_array_for_val_test, prediction_array_for_val_test, possibility_array_for_val_test, total_loss_for_val_test, auc_for_val_test, acc_for_val_test
        return label_array_for_val_test, prediction_array_for_val_test, possibility_array_for_val_test, \
               total_loss_for_val_test, acc_for_val_test, auc_for_val_test, macro_auc_val_test, micro_auc_val_test


class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(sys_time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s

def main(args):

    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)

    loss_fun = torch.nn.CrossEntropyLoss()
    
    
    all_data = joblib.load(args.all_data_path)
    # patient_and_label = joblib.load(args.patient_and_label_path)
    df = pd.read_csv(args.label_path)
    df['label_u'], unique = pd.factorize(df['label'])
    k = [i.replace(".svs","")for i in df["slide_id"].to_list()]
    v = df["label_u"].to_list()
    patient_and_label = dict(zip(k, v))
        
    patiens_list = []
    label_list = []

    for item in patient_and_label:
        patiens_list.append(item)
        label_list.append(patient_and_label[item])

    all_fold_auc = []
    all_fold_acc = []
    all_pre_result = []

    
    since = time.time()
    for repeat_num_temp in range(args.repeat_num):

        pre_result = {}
        fold_auc = []
        fold_acc = []

        seed = args.divide_seed + repeat_num_temp
        setup_seed(repeat_num_temp)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_num = 0
        print('seed', seed, 'repeat_num_temp', repeat_num_temp)
        for train_index, test_index in kf.split(patiens_list, label_list):
            fold_num = fold_num + 1
            print('fold:', fold_num)
            # best_acc_val_fold = 0.0
            best_acc_test_fold = 0.0
            # best_auc_val_fold = 0
            best_auc_test_fold = 0
            # train_index_split, val_index_split = return_train_val_index(list(train_index), label_list, seed=1)
            data_for_train, label_for_train = reuturn_data_label(all_data, list(train_index), patiens_list, label_list,
                                                                 args.batch_size)
            # data_for_val, label_for_val = reuturn_data_label(all_data, val_index_split, patiens_list, label_list,
            #                                                  args.batch_size)
            data_for_test, label_for_test = reuturn_data_label(all_data, list(test_index), patiens_list, label_list,
                                                               args.batch_size)
            train_sample_num = 0
            test_sample_num = 0
            for x in label_for_train:
                train_sample_num = train_sample_num + len(x)
            for x in label_for_test:
                test_sample_num = test_sample_num + len(x)
            print("train_sample_num: " + str(train_sample_num)
                  + " test_sample_num: " + str(test_sample_num))
            
            model = HiGT(
                gcn_in_channels = args.gcn_channel_list[0],
                gcn_hid_channels = args.gcn_channel_list[1],
                gcn_out_channels = args.gcn_channel_list[2],
                gcn_drop_ratio = args.drop_out_ratio,
                mhit_num=args.mhit_num,
                fusion_exp_ratio=args.fusion_exp_ratio,
                out_classes = args.num_classes
            )
            
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(step_size=20, gamma=0.9, optimizer=optimizer)

            for epoch_num in range(args.epochs):
                model.train()
                prediction_array = []
                label_array = []
                possibility_array = []
                total_loss_for_train = 0
                for index_batch_for_train, data_batch_for_train in enumerate(data_for_train):
                    batch_loss = 0
                    label_batch_for_train = label_for_train[index_batch_for_train]
                    for index1, data_train_single in enumerate(data_batch_for_train):
                        label_train_single = label_batch_for_train[index1]
                        label_array.append(label_train_single)
                        # print(data_train_single)
                        # print("=====")
                        data_train_single = data_train_single.to(device)
                        label_train_single = torch.tensor([label_train_single]).to(device)

                        # try:
                        Y_prob = model(data_train_single)
                        # except:
                        #     pass
                        output_for_train = Y_prob
                        prediction_for_train = torch.argmax(Y_prob)
                        loss = loss_fun(output_for_train,F.one_hot(label_train_single, num_classes=args.num_classes).squeeze().float())

                        batch_loss += loss
                        total_loss_for_train += loss
                        prediction_array.append(prediction_for_train.cpu().item())
                        possibility_array.append(Y_prob.squeeze().cpu().detach().numpy())

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                acc = return_acc(prediction_array, label_array)
                auc, macro_auc, micro_auc = return_auc(possibility_array, label_array, args)

                label_array_for_test, prediction_array_for_test, possibility_array_for_test, \
                total_loss_for_test, acc_for_test, auc_for_test, macro_auc_for_test, micro_auc_for_test,= val_test_block(
                    model, loss_fun, device, data_for_test, label_for_test, args)
                scheduler.step()

                if (acc_for_test + micro_auc_for_test) >= (best_acc_test_fold + best_auc_test_fold):
                    best_auc_test_fold = micro_auc_for_test
                    best_acc_test_fold = acc_for_test
                    print("best_acc_plus_auc_test_fold "+str(best_acc_test_fold) + ", save t_model")
                    torch.save(model.state_dict(),
                               f'{args.weight_path}' + f'patient_wise_no_val_fold' + str(fold_num) + '_best.pth')

                print(
                    "epoch: {:2d}, train_loss: {:.4f}, train_acc: {:.4f}, train_micro_auc: {:.4f}, train_macro_auc: {:.4f}, "
                    "test_loss: {:.4f}, test_acc: {:.4f}, test_micro_auc: {:.4f}, test_macro_auc: {:.4f}".format(
                        epoch_num, total_loss_for_train / train_sample_num, acc, micro_auc, macro_auc,
                        total_loss_for_test / test_sample_num, acc_for_test, micro_auc_for_test, macro_auc_for_test,))
                print("train subtype_auc: " + str(auc))
                print("test subtype_auc: " + str(auc_for_test))

                time_elapsed = time.time() - since
                print(
                    'Time completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            model.load_state_dict(torch.load(f'{args.weight_path}' + f'patient_wise_no_val_fold' + str(fold_num) + '_best.pth'))
            model.eval()
            test_pre = []
            test_res = []
            label_of_test = []
            with torch.no_grad():
                for index_batch_for_val_test, data_batch_for_val_test in enumerate(data_for_test):
                    label_batch_for_val_test = label_for_test[index_batch_for_val_test]
                    for index_temp, data_val_test_single in enumerate(data_batch_for_val_test):
                        label_val_test_single = label_batch_for_val_test[index_temp]
                        label_of_test.append(label_val_test_single)
                        data_val_test_single = data_val_test_single.to(device)
                        label_val_test_single = torch.tensor([label_val_test_single]).to(device)
                        # try:
                        Y_prob = model(data_val_test_single['data_id'])
                        # except:
                        #     pass
                        res = Y_prob
                        prediction_for_val_test = torch.argmax(Y_prob)

                        test_pre.append(prediction_for_val_test.cpu().item())
                        test_res.append(res.squeeze().cpu().detach().numpy())
                        pre_result[data_val_test_single['data_id']] = res.cpu().detach().numpy()[0]
                    
            acc_of_test = return_acc(test_pre, label_of_test)
            auc_of_test, macro_auc_for_test, micro_auc_for_test = return_auc(test_res, label_of_test, args)
            precision = precision_score(label_of_test, test_pre, average='weighted')
            confusion = confusion_matrix(label_of_test, test_pre)
            recall = recall_score(label_of_test, test_pre, average='weighted')
            f1 = f1_score(label_of_test, test_pre, average='weighted')
            report = classification_report(label_of_test, test_pre, target_names=['Early', 'Terminal'])

            fold_auc.append(micro_auc_for_test)
            fold_acc.append(acc_of_test)
            
            print('macro_auc: ', macro_auc_for_test)
            print('micro_auc: ', micro_auc_for_test)
            print('subtype_auc: ', auc_for_test)
            print('acc of test: ', acc_of_test)
            print('precision score: ', precision)
            print('recall score: ', recall)
            print('F1 score: ', f1)
            print('confusion matrix: ')
            print(confusion)
            print('report:')
            print(report)

            print("test_predict: "+str(test_pre))
            print("label_of_test: "+str(label_of_test))


        all_pre_result.append(pre_result)
        all_fold_auc.append(fold_auc)
        all_fold_acc.append(fold_acc)
        print('seed', seed)
        print('fold micro auc:', fold_auc, ',mean:', np.mean(fold_auc))
        print('fold acc:', fold_acc, ',mean:', np.mean(fold_acc))

    print('all auc:')
    for r in all_fold_auc:
        print(r)
    print('mean micro auc', np.mean(np.array(all_fold_auc)))
    print('all acc:')
    for r in all_fold_acc:
        print(r)
    print('mean acc', np.mean(np.array(all_fold_acc)))

if __name__ == '__main__':
    try:
        args = get_params()
        main(args)
    except Exception as exception:
        raise