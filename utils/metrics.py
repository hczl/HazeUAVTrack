# 创建一个 metrics.py 模块，包含 mAP、F1、MOTA、MOTP、ID切换计算函数
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def filter_ignore(preds, ignores):
    result = []
    for box in preds:
        if all(iou(box[:4], ig) < 0.5 for ig in ignores):
            result.append(box)
    return result

def compute_map(preds_all, gts_all, ignore_masks=None, iou_threshold=0.5):
    tp, fp, total_gt = 0, 0, 0
    for preds, gts, ignores in zip(preds_all, gts_all, ignore_masks or [[]]*len(preds_all)):
        preds = sorted(filter_ignore(preds, ignores), key=lambda x: -x[4])
        matched = set()
        total_gt += len(gts)
        for pred in preds:
            matched_flag = False
            for i, gt in enumerate(gts):
                if i in matched:
                    continue
                if iou(pred[:4], gt[:4]) >= iou_threshold:
                    matched.add(i)
                    matched_flag = True
                    break
            if matched_flag:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def compute_f1(preds_all, gts_all, ignore_masks=None, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    for preds, gts, ignores in zip(preds_all, gts_all, ignore_masks or [[]]*len(preds_all)):
        preds = filter_ignore(preds, ignores)
        matched = set()
        for pred in preds:
            match = -1
            for i, gt in enumerate(gts):
                if i in matched:
                    continue
                if iou(pred[:4], gt[:4]) >= iou_threshold:
                    match = i
                    break
            if match >= 0:
                matched.add(match)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(matched)
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom else 0.0

def compute_mota(preds_all, gts_all, ignore_masks=None, iou_threshold=0.5):
    misses, false_positives, id_switches, matches_total = 0, 0, 0, 0
    last_match_ids = {}
    total_gt = 0
    total_iou = 0

    for t, (preds, gts, ignores) in enumerate(zip(preds_all, gts_all, ignore_masks or [[]]*len(preds_all))):
        preds = filter_ignore(preds, ignores)
        total_gt += len(gts)
        cost_matrix = np.zeros((len(gts), len(preds)), dtype=np.float32)
        for i, gt in enumerate(gts):
            for j, pred in enumerate(preds):
                cost_matrix[i, j] = 1 - iou(gt[:4], pred[:4])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_ids = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > 1 - iou_threshold:
                continue
            gt_id = gts[r][4]
            matched_ids[gt_id] = c
            matches_total += 1
            total_iou += 1 - cost_matrix[r, c]
            if gt_id in last_match_ids and last_match_ids[gt_id] != c:
                id_switches += 1
            last_match_ids[gt_id] = c
        misses += len(gts) - len(matched_ids)
        false_positives += len(preds) - len(matched_ids)
    mota = 1 - (misses + false_positives + id_switches) / total_gt if total_gt else 0.0
    motp = total_iou / matches_total if matches_total else 0.0
    return mota, motp, id_switches
