import math
import torch
import logging
import numpy as np

import torch.nn.functional as F

from tqdm import tqdm

from utils import split_cluster_acc_v1, split_cluster_acc_v2, get_dis_max, compute_d_intra, compute_anmi, SmoothedValue, MetricLogger


def cos_eps_loss(u, y, hash_center):

    u_norm = F.normalize(u)
    centers_norm = F.normalize(hash_center)
    cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class

    loss = torch.nn.CrossEntropyLoss()(cos_sim, y)

    return loss

def sep_loss(protop_centers, L = 12, dis_max = 3, alpha=0.95):
    labels = torch.arange(protop_centers.shape[0])
    dot_product = torch.matmul(protop_centers, protop_centers.T)
    hamming_distance = 0.5 * (L*alpha - dot_product)
    mask_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    mask_diff = mask_diff.cuda()
    loss_sep = (F.relu(dis_max - hamming_distance) * mask_diff.float()).sum(-1)
    return loss_sep.mean()


def train_and_evaluate(model, 
                    data_loader, 
                    test_loader_unlabelled, 
                    optimizer, device, 
                    epoch, loss_scaler, 
                    max_norm, 
                    model_ema, 
                    args=None, 
                    set_training_mode=True):
    
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    dis_max = get_dis_max(args)

    for batch_index, (samples, targets, _, ind) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, hash_feat = model(samples)

            # supervise loss
            outputs = F.log_softmax(outputs, dim=1)
            loss_protop = torch.nn.NLLLoss()(outputs, targets)

            ## get hash centers
            samples_per_class = args.global_proto_per_class
            class_means = torch.stack([model.prototype_vectors_global[i:i+samples_per_class].mean(0) for i in range(0, model.prototype_vectors_global.size(0), samples_per_class)])
            hash_centers = model.hash_head(class_means)

            hash_centers_sign = torch.nn.Tanh()(hash_centers*3)

            # hash centers separation loss
            loss_sep = sep_loss(hash_centers_sign, L=args.hash_code_length, dis_max=dis_max)

            # hash center quantization loss
            loss_quan = (1 - torch.abs(hash_centers_sign)).mean() 

            ## hash feature optimize loss
            loss_feature = cos_eps_loss(hash_feat, targets, hash_centers) 
            
            loss = loss_protop + loss_sep * args.alpha + loss_quan * args.alpha + loss_feature * args.beta

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # metric_logger.update(loss=loss_value)
        metric_logger.update(loss_protop=loss_protop.item())
        metric_logger.update(loss_feature=loss_feature.item())
        metric_logger.update(loss_sep=loss_sep.item())        
        metric_logger.update(loss_quan=loss_quan.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluate(test_loader=test_loader_unlabelled, model=model, args=args, centers=hash_centers.cpu().sign(), dis_max=dis_max)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def hash_assign(feats_hash, centers, radius, device='cuda'):
    """
    Greedy on-the-fly assignment of hash codes to categories.
    Samples are visited in order; each one joins the nearest existing category
    (first index on ties) if its Hamming distance is <= radius, otherwise it
    opens a new category with its own code. An exact match has distance 0,
    so it always joins the first identical code.

    # Arguments
        feats_hash: binary codes in {0, 1}, tensor with shape `(n_samples, L)`
        centers: initial category codes (sign of hash centers), tensor with shape `(n_classes, L)`
        radius: Hamming sphere radius

    # Return
        preds: category index of each sample, numpy.array with shape `(n_samples,)`
    """
    feats_hash = feats_hash.to(device=device, dtype=torch.int8)
    n_init, n_samples = centers.shape[0], feats_hash.shape[0]
    # Preallocate room for every sample opening a new category
    bank = torch.empty(n_init + n_samples, feats_hash.shape[1], dtype=torch.int8, device=device)
    bank[:n_init] = centers.to(device=device, dtype=torch.int8)
    n_cat = n_init

    preds = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        feat = feats_hash[i]
        distances = (bank[:n_cat] != feat).sum(1)
        min_index = torch.argmin(distances)  # first minimum, same as list.index(min(...))
        min_index, min_distance = torch.stack([min_index, distances[min_index]]).tolist()

        if min_distance <= radius:
            preds[i] = min_index
        else:
            # Outside every Hamming sphere: create a new category
            bank[n_cat] = feat
            preds[i] = n_cat
            n_cat += 1

    return preds

@torch.no_grad()
def evaluate(test_loader, model, args, centers, dis_max):
    radius = max(math.floor(dis_max / 2), 1)
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    logger.info(f"Radius: {radius}")

    metric_logger = MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()
        label = label.cuda()
        feats = model(images)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
 
        mask = np.append(mask, np.array([True if x.item() in range(args.labeled_nums) else False for x in label]))
    
    all_feats = np.concatenate(all_feats, axis=0)

    # Hash
    feats_hash = torch.from_numpy(all_feats > 0)

    # Store the category index corresponding to each feature
    preds1 = hash_assign(feats_hash, centers, radius)

    all_acc, old_acc, new_acc = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask)
    logger.info(f'test len(list(set(preds1))): {len(list(set(preds1)))} len(preds): {len(preds1)}')
    logger.info(f"Evaluate V1: all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
    logger.info(f"Evaluate V2: all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # Code quality metrics on binary hash codes (d_intra lower is better, anMI higher is better)
    codes = (all_feats > 0).astype(np.int8)
    mask = mask.astype(bool)
    for name, m in [('all', np.ones_like(mask)), ('old', mask), ('new', ~mask)]:
        if not m.any():
            continue
        d_intra = compute_d_intra(codes[m], targets[m])
        anmi = compute_anmi(codes[m], targets[m])
        logger.info(f"Code {name}: d_intra: {d_intra:.4f} anMI: {anmi:.4f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}