import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
# from models.HIGT.higt import HIGT
from models.MulGT.MulGT import MulGT

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
        
    #=====
    elif args.model_type == "HiGT":
        model = HIGT(gcn_in_channels=1024, gcn_hid_channels=1024, gcn_out_channels=1024, gcn_drop_ratio=0.3, patch_ratio=4,
            pool_ratio=[0.5,5], re_patch_size=64, out_classes=args.n_classes, mhit_num=3, fusion_exp_ratio=4
        )
    
    elif args.model_type == "MulGT":
        if args.mulgt_task_type == "multi":
            model = MulGT(args.n_classes[0], args.n_classes[1], input_dim=args.input_dim)
        else:
            model = MulGT(args.n_classes, args.n_classes, input_dim=args.input_dim)

    #=====
    elif args.model_type == 'mil':
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    # model.relocate()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print('Done!')
    # print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')

    train_loader = torch.utils.data.DataLoader(dataset=train_split, batch_size=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_split, batch_size=1)
    test_loader = torch.utils.data.DataLoader(dataset=test_split, batch_size=1)
    # train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    # val_loader = get_split_loader(val_split,  testing = args.testing)
    # test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                                early_stopping, writer, loss_fn, args.results_dir)
            
        elif args.model_type == "MulGT":
            train_loop_mulgt(epoch, model, train_loader, optimizer, args.typing_n_classes, args.stage_n_classes, 
                             args.mulgt_task_type, args.grad_norm, args.mulgt_pool_method, writer, loss_fn)
            stop = validate_mulgt(cur, epoch, model, val_loader, args.typing_n_classes, args.stage_n_classes, 
                                args.mulgt_task_type, args.grad_norm, args.mulgt_pool_method, early_stopping, writer, loss_fn, args.results_dir)
            
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if args.model_type == "MulGT":
        
        typing_val_error, stage_val_error, typing_auc, stage_auc, typing_acc_logger, stage_acc_logger = summary(model, val_loader, args.n_classes)
        print('Val subtype_error: {:.4f}, stage_error: {:.4f}, ROC subtype_AUC: {:.4f}, stage_AUC: {:.4f}'.format(typing_val_error, stage_val_error, typing_auc, stage_auc))

        typing_test_error, stage_test_error, typing_auc, stage_auc, typing_acc_logger, stage_acc_logger = summary(model, test_loader, args.n_classes)
        print('Test subtype_error: {:.4f}, stage_error: {:.4f}, ROC subtype_AUC: {:.4f}, stage_AUC: {:.4f}'.format(typing_test_error, stage_test_error, typing_auc, stage_auc))

        for i in range(args.typing_n_classes):
            acc, correct, count = typing_acc_logger.get_summary(i)
            print('class(subtype) {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

        for i in range(args.stage_n_classes):
            acc, correct, count = stage_acc_logger.get_summary(i)
            print('class(subtype) {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    else:
        _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop_re(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, label, _ = model(data, label)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def train_loop_mulgt(epoch, model, loader, optimizer, typing_n_classes, stage_n_classes, mulgt_task_type, grad_norm, mulgt_pool_method="dense_diff_pool", writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    print("training!!!!!")
    
    train_loss = 0.
    
    typing_acc_logger = Accuracy_Logger(n_classes=typing_n_classes)
    stage_acc_logger = Accuracy_Logger(n_classes=stage_n_classes)
    train_typing_error = 0.
    train_stage_error = 0.

    print('\n')
    batch_loss = 0.
    for batch_idx, (data, label) in enumerate(loader):
        try:
            typing_prob, typing_preds, typing_labels, typing_loss, \
            stage_prob, stage_preds, stage_labels, stage_loss, \
            reg_loss = model(data, label)
        except:
            continue

        if mulgt_task_type == "multi":
            if grad_norm:
                stage_loss = stage_loss.unsqueeze(0)
                typing_loss = typing_loss.unsqueeze(0)
                loss = torch.cat([stage_loss, typing_loss],0)
                loss = grad_norm(loss)/2.0
                stage_loss = stage_loss.squeeze(0)
                typing_loss = typing_loss.squeeze(0)
            else:
                loss = (stage_loss + typing_loss)/2.0

        elif mulgt_task_type == "subtype":
            loss = typing_loss
        elif mulgt_task_type == "stage":
            loss = stage_loss

        if mulgt_pool_method == "dense_diff_pool" or mulgt_pool_method == "dense_mincut_pool" or reg_loss:
            loss = loss + reg_loss

        loss = loss.mean()
        batch_loss += loss

        

        typing_acc_logger.log(typing_preds, typing_labels)
        stage_acc_logger.log(stage_preds, stage_labels)

        loss_value = loss.item()
        # train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {} and {}, bag_size: {}'.format(batch_idx, loss_value, typing_labels.item(), stage_labels.item(), data.size(0)))
        

        typing_error = calculate_error(typing_preds, typing_labels)
        train_typing_error += typing_error

        stage_error = calculate_error(stage_preds, stage_labels)
        train_stage_error += stage_error

        if batch_idx % 1 == 0 or batch_idx == len(loader) - 1:
            batch_loss = batch_loss/1
            optimizer.zero_grad()

            if grad_norm:
                batch_loss.backward(retain_graph=True)
                grad_norm.additional_forward_and_backward(grad_norm_weights=model.W_O, total_loss=batch_loss)
            else:
                batch_loss.backward()

            optimizer.step()
            batch_loss = 0

            torch.cuda.empty_cache()

        train_loss += loss
    # calculate loss and error for epoch
    train_loss /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_stage_error: {:.4f}'.format(epoch, train_loss, train_stage_error))
    print('Epoch: {}, train_loss: {:.4f}, train_typing_error: {:.4f}'.format(epoch, train_loss, train_typing_error))
    for i in range(typing_n_classes):
        typing_acc, typing_correct, typing_count = typing_acc_logger.get_summary(i)
        print('typing class {}: acc {}, correct {}/{}'.format(i, typing_acc, typing_correct, typing_count))
        if writer:
            writer.add_scalar('train/typing_class_{}_acc'.format(i), typing_acc, epoch)

    for i in range(stage_n_classes):
        stage_acc, stage_correct, stage_count = stage_acc_logger.get_summary(i)
        print('stage class {}: acc {}, correct {}/{}'.format(i, stage_acc, stage_correct, stage_count))
        if writer:
            writer.add_scalar('train/stage_class_{}_acc'.format(i), stage_acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/stage_error', train_stage_error, epoch)
        writer.add_scalar('train/typing_error', train_typing_error, epoch)

def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False
   
# 改一下model的逻辑
'''
所有数据直接传到model里面进行处理，这样train函数就会方便很多
'''
def validate_mulgt(cur, epoch, model, loader, typing_n_classes, stage_n_classes, mulgt_task_type, grad_norm, mulgt_pool_method="dense_diff_pool", early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    typing_acc_logger = Accuracy_Logger(n_classes=typing_n_classes)
    stage_acc_logger = Accuracy_Logger(n_classes=stage_n_classes)
    val_typing_error = 0.
    val_stage_error = 0.
    # loader.dataset.update_mode(True)
    val_loss = 0.
    
    typing_probs = np.zeros((len(loader), typing_n_classes))
    typing_labels = np.zeros(len(loader))

    stage_probs = np.zeros((len(loader), typing_n_classes))
    stage_labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            # data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            try:
                typing_prob, typing_hat, typing_label, typing_loss, \
                stage_prob, stage_hat, stage_label, stage_loss, \
                reg_loss = model(data, label)
            except:
                continue

            typing_acc_logger.log(typing_hat, typing_label)
            stage_acc_logger.log(stage_hat, stage_label)
            

            typing_probs[batch_idx] = typing_prob.cpu().numpy()
            typing_labels[batch_idx] = typing_label.item()
            stage_probs[batch_idx] = stage_prob.cpu().numpy()
            stage_labels[batch_idx] = stage_label.item()
            

            if mulgt_task_type == "multi":
                if grad_norm:
                    stage_loss = stage_loss.unsqueeze(0)
                    typing_loss = typing_loss.unsqueeze(0)
                    loss = torch.cat([stage_loss, typing_loss],0)
                    loss = grad_norm(loss)/2.0
                    stage_loss = stage_loss.squeeze(0)
                    typing_loss = typing_loss.squeeze(0)
                else:
                    loss = (stage_loss + typing_loss)/2.0

            elif mulgt_task_type == "subtype":
                loss = typing_loss
            elif mulgt_task_type == "stage":
                loss = stage_loss

            if mulgt_pool_method == "dense_diff_pool" or mulgt_pool_method == "dense_mincut_pool" or reg_loss:
                loss = loss + reg_loss

            val_loss += loss.item()
            error = calculate_error(typing_hat, typing_label)
            val_typing_error += error

            error = calculate_error(stage_hat, stage_label)
            val_stage_error += error

    val_stage_error /= len(loader)
    val_typing_error /= len(loader)
    val_loss /= len(loader)

    if typing_n_classes == 2:
        typing_auc = roc_auc_score(typing_labels, typing_probs[:, 1])
    else:
        typing_auc = roc_auc_score(typing_labels, typing_probs, multi_class='ovr')
    if stage_n_classes == 2:

        stage_auc = roc_auc_score(stage_labels, stage_probs[:, 1])
    else:
        stage_auc = roc_auc_score(stage_labels, stage_probs, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/typing_auc', typing_auc, epoch)
        writer.add_scalar('val/stage_auc', stage_auc, epoch)
        writer.add_scalar('val/typing_error', val_typing_error, epoch)
        writer.add_scalar('val/stage_error', val_stage_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, typing_val_error: {:.4f}, stage_val_error: {:.4f}, typing_auc: {:.4f}, stage_auc: {:.4f}'.format(val_loss, val_typing_error, val_stage_error, typing_auc, stage_auc))
    for i in range(typing_n_classes):
        acc, correct, count = typing_acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    for i in range(stage_n_classes):
        acc, correct, count = stage_acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger

def summary_mulgt(model, loader, typing_n_classes, stage_n_classes, mulgt_task_type):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    typing_acc_logger = Accuracy_Logger(n_classes=typing_n_classes)
    stage_acc_logger = Accuracy_Logger(n_classes=stage_n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    typing_all_probs = np.zeros((len(loader), typing_n_classes))
    typing_all_labels = np.zeros(len(loader))
    stage_all_probs = np.zeros((len(loader), stage_n_classes))
    stage_all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            try:
                typing_prob, typing_hat, typing_label, typing_loss, \
                stage_prob, stage_hat, stage_label, stage_loss, \
                reg_loss = model(data, label)
            except:
                continue

        typing_acc_logger.log(typing_hat, typing_label)
        typing_probs = typing_prob.cpu().numpy()
        typing_all_probs[batch_idx] = typing_probs
        typing_all_labels[batch_idx] = typing_label.item()

        stage_acc_logger.log(stage_hat, stage_label)
        stage_probs = stage_prob.cpu().numpy()
        stage_all_probs[batch_idx] = stage_probs
        stage_all_labels[batch_idx] = stage_label.item()
        
        if mulgt_task_type == "multi":
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'typing_prob': typing_probs, 'typing_label': typing_label.item(), 'stage_prob': stage_probs, 'stage_label': stage_label.item()}})
        elif mulgt_task_type == "subtype":
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'typing_prob': typing_probs, 'typing_label': typing_label.item()}})
        elif mulgt_task_type == "stage":
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'stage_prob': stage_probs, 'stage_label': stage_label.item()}})

        typing_error = calculate_error(typing_hat, label)
        typing_test_error += typing_error
        stage_error = calculate_error(stage_hat, label)
        stage_test_error += stage_error

    typing_test_error /= len(loader)
    stage_test_error /= len(loader)


    if typing_n_classes == 2:
        typing_auc = roc_auc_score(typing_all_labels, typing_all_probs[:, 1])
        typing_aucs = []
    else:
        typing_aucs = []
        binary_labels = label_binarize(typing_all_labels, classes=[i for i in range(typing_n_classes)])
        for class_idx in range(typing_n_classes):
            if class_idx in typing_all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], typing_all_probs[:, class_idx])
                typing_aucs.append(calc_auc(fpr, tpr))
            else:
                typing_aucs.append(float('nan'))

        typing_auc = np.nanmean(np.array(typing_aucs))

    if stage_n_classes == 2:
        stage_auc = roc_auc_score(stage_all_labels, stage_all_probs[:, 1])
        stage_aucs = []
    else:
        stage_aucs = []
        binary_labels = label_binarize(stage_all_labels, classes=[i for i in range(stage_n_classes)])
        for class_idx in range(stage_n_classes):
            if class_idx in stage_all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], stage_all_probs[:, class_idx])
                stage_aucs.append(calc_auc(fpr, tpr))
            else:
                stage_aucs.append(float('nan'))

        stage_auc = np.nanmean(np.array(stage_aucs))


    return patient_results, typing_test_error, stage_test_error, typing_auc, stage_auc, typing_acc_logger, stage_acc_logger
