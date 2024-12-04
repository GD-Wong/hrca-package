from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Warning:Tensorboard may be not installed with pytorch")
    pass
import scanpy as sc
import numpy as np
import pandas as pd
import os
import datetime
import sys
import timeit
import gc
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from . import dataload
from . import model
from . import utils

matplotlib.use('Agg')


def pretrain(pre_train_adata, save_path, dropout=0.7, n_epochs=2000, epoch=0, batch_size=128, n_cpu=1, head_num=8,
              n_layers=8, lr=1e-3, decay_epoch=200, lr_decrease_factor=0.1, lr_decrease_patience=50, b1=0.9, b2=0.999,
              checkpoint_interval=100, tb_writer=False, use_device=0):
    dataloader = DataLoader(
        dataload.AnnDataLoader(adata=pre_train_adata),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu
    )
    genes = pre_train_adata.var.index
    max_step = n_epochs * len(dataloader)
    os.makedirs("%s/pretrain_model/" % save_path, exist_ok=True)
    np.save("%s/features.npy" % save_path, list(genes))
    input_shape = dataloader.dataset.input.shape[1]
    module = model.Gene_AE(input_shape, head_number=head_num, n_layers=n_layers)
    criterion = torch.nn.MSELoss()
    if use_device != "cpu" and torch.cuda.is_available():
        module = module.cuda(device=use_device)
        criterion = criterion.cuda(device=use_device)
    if epoch != 0:
        resume_point = torch.load("%s/checkpoint" % save_path)
        module.load_state_dict(torch.load("%s/pretrain_model/Gene_AE_%d.pth" % (save_path, epoch)))
    else:
        module.apply(utils.weights_init_normal)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, betas=(b1, b2))
    if epoch != 0:
        optimizer.load_state_dict(resume_point['optimizer'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_decrease_factor,
                                                              patience=lr_decrease_patience, eps=1e-8, cooldown=0)
    prev_time = timeit.default_timer()
    if tb_writer:
        writer = SummaryWriter("%s/pretrain_model/" % save_path)
    step = epoch * len(dataloader)
    beta = 0.98
    avg_loss = 0.
    for epoch in range(epoch + 1, n_epochs + 1):
        epoch_sum_loss = 0.
        for i, batch in enumerate(dataloader):
            step += 1
            # count_data，还原目标为norm_data,输入dropout后标准化了的input_data作为训练输入
            count_data = Variable(batch).to(next(module.parameters()).device)
            norm_data = utils.normalize_torch(count_data, scalefactor=10000, log_flag=True)
            input_data = utils.dropout(count_data, dropout_flag=True, dropout_rate=dropout, norm_flag=True)
            module.train()
            loss = criterion(module(input_data), norm_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 平滑化损失
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** step)
            # 统计本轮loss总值
            epoch_sum_loss += smoothed_loss
            if tb_writer:
                writer.add_scalar("loss/smoothed loss", smoothed_loss, step)
                writer.add_scalar("loss/true loss", loss, step)
                writer.add_scalar("learning rate", optimizer.state_dict()['param_groups'][0]['lr'], step)
            batches_left = max_step - step
            time_left = datetime.timedelta(seconds=batches_left * (timeit.default_timer() - prev_time))
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %g] [lr: %g]ETA: %s \n"
                % (
                    epoch,
                    n_epochs,
                    i + 1,
                    len(dataloader),
                    smoothed_loss,
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    time_left,
                )
            )
            sys.stdout.flush()
            prev_time = timeit.default_timer()
        # 损失不下降则降低学习率
        if decay_epoch != -1 and epoch > decay_epoch:
            epoch_avg_loss = epoch_sum_loss / len(dataloader)
            lr_scheduler.step(epoch_avg_loss)
        # 检查点轮次进行模型记录和测试
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # 保存模型参数
            torch.save(module.state_dict(), "%s/pretrain_model/Gene_AE_%d.pth" % (save_path, epoch))
            # 记录检查点状态
            checkpoint = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, "%s/checkpoint" % save_path)
    if tb_writer:
        writer.close()
    return module


def fine_tune(train_adata, test_adata, AE_module, save_path, obs_name="cell_type", dropout=0.7, n_epochs=200, batch_size=128, n_cpu=1,
              n_layers=3, smoothing_factor=0.1, temperature=1.0, lr=1e-3, decay_epoch=20, lr_decrease_factor=0.1, lr_decrease_patience=25, b1=0.9, b2=0.999,
              checkpoint_interval=50, tb_writer=False, use_device=0):
    data_loader = DataLoader(
        dataload.AnnDataLoader(adata=train_adata, obs_name=obs_name, return_label=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu
    )
    test_loader = DataLoader(
        dataload.InputLoader(input_adata=test_adata, ref_adata=train_adata, norm_flag=True, dropout_flag=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu
    )
    max_step = n_epochs * len(data_loader)
    step = 0
    os.makedirs("%s/fine_tune_model/" % save_path, exist_ok=True)
    # 保存字典
    dictionary = data_loader.dataset.dict
    np.save("%s/dictionary.npy" % save_path, dictionary)
    # 注释模型
    cell_type_num = set(data_loader.dataset.label).__len__()
    predictor = model.SE_Classifier(output_shape=cell_type_num, input_shape=1024, n_layers=n_layers)
    predictor.apply(utils.weights_init_normal)
    reconstruct_criterion = torch.nn.MSELoss()
    classify_criterion = utils.LabelSmoothingLoss(classes=cell_type_num, smoothing=smoothing_factor, temperature=temperature)
    if use_device !="cpu" and torch.cuda.is_available():
        AE_module = AE_module.cuda(device=use_device)
        predictor = predictor.cuda(device=use_device)
        reconstruct_criterion = reconstruct_criterion.cuda(device=use_device)
        classify_criterion = classify_criterion.cuda(device=use_device)
    optimizer = torch.optim.Adam(list(AE_module.parameters()) + list(predictor.parameters()), lr=lr,
                                 betas=(b1, b2))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_decrease_factor, eps=1e-8, patience=lr_decrease_patience)
    prev_time = timeit.default_timer()
    if tb_writer:
        writer = SummaryWriter("%s/fine_tune_model/" % save_path)
    beta = 0.98
    avg_classify_loss = 0.
    avg_reconstruct_loss = 0.
    avg_loss = 0.

    for epoch in range(1, n_epochs + 1):
        gc.collect()
        torch.cuda.empty_cache()
        epoch_sum_loss = 0.
        sum_prec = 0.
        for i, (batch, label) in enumerate(data_loader):
            step += 1
            # count_data，还原目标为norm_data,输入dropout后标准化了的source_input作为训练输入
            count_data = Variable(batch).to(next(AE_module.parameters()).device)
            norm_data = utils.normalize_torch(count_data, scalefactor=10000, log_flag=True)
            source_input = utils.dropout(count_data, dropout_flag=True, dropout_rate=dropout, norm_flag=True)
            label_int = torch.squeeze(torch.LongTensor(label).to(next(AE_module.parameters()).device))
            predictor.train()
            reconstruct_loss = reconstruct_criterion(AE_module(source_input), norm_data)
            prob = predictor(AE_module.encode(source_input))
            classify_loss = classify_criterion(prob, label_int)
            prec = (prob.argmax(dim=1) == label_int).sum() / len(label_int)
            sum_prec += prec
            loss = 0.1 * reconstruct_loss + 0.9 * classify_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches_left = max_step - step
            time_left = datetime.timedelta(seconds=batches_left * (timeit.default_timer() - prev_time))
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %g rec_loss: %g cls_loss: %g][precision: %g] [lr: %g]ETA: %s \n"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(data_loader),
                    loss,
                    reconstruct_loss,
                    classify_loss,
                    prec,
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    time_left,
                )
            )
            sys.stdout.flush()
            # 损失平滑化及其统计
            avg_reconstruct_loss = beta * avg_reconstruct_loss + (1 - beta) * reconstruct_loss.item()
            smoothed_reconstruct_loss = avg_reconstruct_loss / (1 - beta ** step)
            avg_classify_loss = beta * avg_classify_loss + (1 - beta) * classify_loss.item()
            smoothed_classify_loss = avg_classify_loss / (1 - beta ** step)
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** step)
            epoch_sum_loss += smoothed_loss
            if tb_writer:
                writer.add_scalar("smoothed loss/reconstruct", smoothed_reconstruct_loss, step)
                writer.add_scalar("smoothed loss/classify", smoothed_classify_loss, step)
                writer.add_scalar("smoothed loss/loss", smoothed_loss, step)
                writer.add_scalar("true loss/reconstruct", reconstruct_loss, step)
                writer.add_scalar("true loss/classify", classify_loss, step)
                writer.add_scalar("true loss/loss", loss, step)
                writer.add_scalar("learning rate", optimizer.state_dict()['param_groups'][0]['lr'], step)
            prev_time = timeit.default_timer()
        if tb_writer:
            writer.add_scalar("train_precision", sum_prec / len(data_loader), epoch)
        if decay_epoch != -1 and epoch > decay_epoch:
            epoch_avg_loss = epoch_sum_loss / len(data_loader)
            lr_scheduler.step(epoch_avg_loss)
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            torch.save(AE_module.state_dict(), "%s/fine_tune_model/Gene_AE_%d.pth" % (save_path, epoch))
            torch.save(predictor.state_dict(),"%s/fine_tune_model/SE_Classifier_%d.pth" % (save_path, epoch))
            precision_test, cm_heatmap_test = test(test_loader, AE_module, predictor, dictionary)
            if tb_writer:
                writer.add_scalar("precision/test", precision_test, epoch)
                writer.add_figure("test/confusion_heatmap", cm_heatmap_test, epoch)
            cm_heatmap_test.savefig("%s/fine_tune_model/precision_heatmap_%d.png" % (save_path, epoch), dpi=300, bbox_inches='tight')
            sys.stdout.write("Test data precision: %d" % precision_test)
    if tb_writer:
        writer.close()
    return AE_module,predictor


def test(input_dataloader, encoder, classifier, l_dict):
    encoder.eval()
    classifier.eval()
    all_pred = torch.tensor([], dtype=int)
    for i, source in enumerate(input_dataloader):
        source_input = Variable(source).to(next(encoder.parameters()).device)
        probs = classifier(encoder.encode(source_input))
        pred = probs.argmax(dim=1)
        pred_cpu = pred.cpu()
        all_pred = torch.hstack((all_pred, pred_cpu))
    test_label = np.array(input_dataloader.dataset.obs.cell_type)
    pred_result = np.array(utils.label_int_to_str(all_pred, l_dict))
    #label_lst = ["Naive B", "Memory B", "Follicular B", "Plasma",
    #             "PCV", "Arterial", "Angiogenic", "Lymphatic End", "Capillary",
    #             "Epi", "myCAF", "iCAF", "Pericyte",
    #             "Monocyte", "TAM", "cDC2", "cDC1", "pDC", "mature cDC", "Mast", "Neutrophil",
    #             "CD8+Tem", "CD8+Tcm", "CD8+Tn", "CD8+Tex", "CD8+Tc",
    #             "CD4+Tem", "CD4+Tcm", "CD4+Tn", "CD4+Tex", "Treg", "Th",
    #             "NK", "NKT", "gamma delta T", "MAIT"]
    label_lst = list(l_dict)
    prec = (test_label == pred_result).sum() / len(test_label)
    cm = confusion_matrix(y_true=list(test_label), y_pred=list(pred_result), labels=label_lst, normalize='true')
    fig = plt.figure(figsize=(10, 10), dpi=200)
    sns.heatmap(cm, fmt='.2g', cmap="Blues", annot=True, cbar=False, xticklabels=label_lst, yticklabels=label_lst,
                annot_kws={"fontsize": 4})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return prec, fig

def annotate(adata, features, encoder, classifier, l_dict, result_name = "predicted", bs = 128, workers = 4):
    encoder.eval()
    classifier.eval()
    ref_adata = sc.AnnData(X=np.ones([1,features.__len__()]), var=pd.DataFrame(index=features))
    genes = list(features)
    all_adata = sc.concat({"input": adata, "ref": ref_adata}, label="data_type", join="outer", axis=0)[:, genes]
    input_adata = all_adata[all_adata.obs.data_type == "input", :]
    sc.pp.filter_cells(input_adata, min_genes=80)
    input_dataloader = DataLoader(
        dataload.AdataLoader(adata=input_adata, norm_flag=True),
        batch_size=bs,
        shuffle=False,
        num_workers=workers
    )
    all_pred = torch.tensor([], dtype=int)
    for i, source in enumerate(input_dataloader):
        source_input = Variable(source).to(next(encoder.parameters()).device)
        probs = classifier(encoder.encode(source_input))
        pred = probs.argmax(dim=1)
        pred_cpu = pred.cpu()
        all_pred = torch.hstack((all_pred, pred_cpu))
    pred_result = pd.DataFrame(utils.label_int_to_str(all_pred, l_dict), columns=[result_name], index=input_dataloader.dataset.obs.index)
    return pred_result