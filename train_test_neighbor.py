import time

import torch
import torch.optim as optim
from tqdm import trange
import copy

def build_optimizer(args, params):

    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def train(loader, test_loader, model, loss_function, args, evaluator=None):

    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    start = time.time()
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask].squeeze()
            loss = loss_function(pred, label)
            loss.backward()
            opt.step()
            if len(loader.dataset) == 1:
                total_loss += loss.item() * batch.num_graphs
            else:
                total_loss += loss.item()
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
          test_acc = test(test_loader, model, is_validation=True, evaluator=evaluator)
          test_accs.append(test_acc)
          if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])
        end = time.time()
    final_model = model
    training_time = end-start
    return test_accs, losses, best_model, final_model, best_acc, test_loader, training_time

def test(loader, test_model, is_validation=False, save_model_preds=False, evaluator=None):
    test_model.eval()
    
    if evaluator is None:
        correct = 0
        if save_model_preds:
          print ("Saving Model Predictions for Model Type", test_model.type)
        
        for data in loader:
            
            with torch.no_grad():
                # max(dim=1) returns values, indices tuple; only need indices
                pred = test_model(data).max(dim=1)[1]
                label = data.y
    
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = label[mask]
    
            if save_model_preds:
    
              data_save = {}
              data_save['pred'] = pred.view(-1).cpu().detach().numpy()
              data_save['label'] = label.view(-1).cpu().detach().numpy()

            correct += pred.eq(label).sum().item()
    
        total = 0
        for data in loader:
            total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
        return correct / total
    
    else:
        data = loader.data
        out = test_model(data)
        y_pred = out.argmax(dim=-1, keepdim=True)
    
        if is_validation:
            acc = evaluator.eval({
                'y_true': data.y[data.val_mask.squeeze()],
                'y_pred': y_pred[data.val_mask.squeeze()],
            })['acc']
        else:
            acc = evaluator.eval({
                'y_true': data.y[data.test_mask.squeeze()],
                'y_pred': y_pred[data.test_mask.squeeze()],
            })['acc']
    
        return acc
