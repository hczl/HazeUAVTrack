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

# ---------- 共用索引 ----------
CONF_IDX = 5      # 置信度
ID_IDX   = 4      # track_id
GT_ID_IDX = 4     # GT 第 5 列是目标 ID（若没有，可自行生成）

# ---------- mAP ----------
def compute_map(preds_all, gts_all, ignore_masks=None, iou_threshold=0.5):
    tp, fp, total_gt = 0, 0, 0
    for preds, gts, ignores in zip(preds_all,
                                   gts_all,
                                   ignore_masks or [[]]*len(preds_all)):

        # 过滤忽略区域并按置信度排序
        preds = sorted(filter_ignore(preds, ignores),
                       key=lambda x: -x[CONF_IDX])

        matched = set()
        total_gt += len(gts)
        for pred in preds:
            hit = False
            for i, gt in enumerate(gts):
                if i in matched:
                    continue
                if iou(pred[:4], gt[:4]) >= iou_threshold:
                    matched.add(i)
                    hit = True
                    break
            if hit: tp += 1
            else:   fp += 1
    return tp / (tp + fp) if (tp + fp) else 0.


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

# ---------- MOTA / MOTP / ID Switch ----------
from scipy.optimize import linear_sum_assignment
def compute_mota(preds_all, gts_all, ignore_masks=None, iou_threshold=0.5):
    miss, fp, id_sw, tot_gt = 0, 0, 0, 0
    tot_iou, match_cnt = 0., 0
    last_match = {}        # {gt_id: track_id}

    for preds, gts, ignores in zip(preds_all,
                                   gts_all,
                                   ignore_masks or [[]]*len(preds_all)):

        preds = filter_ignore(preds, ignores)
        tot_gt += len(gts)

        if not preds or not gts:
            miss += len(gts)
            fp   += len(preds)
            continue

        cost = np.ones((len(gts), len(preds)), dtype=np.float32)
        for i, gt in enumerate(gts):
            for j, pr in enumerate(preds):
                cost[i, j] = 1 - iou(gt[:4], pr[:4])

        r, c = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pr = set()
        for gi, pj in zip(r, c):
            if cost[gi, pj] > 1 - iou_threshold:
                continue                              # IoU 不够
            gt_id  = gts[gi][GT_ID_IDX]
            trk_id = preds[pj][ID_IDX]
            matched_gt.add(gi)
            matched_pr.add(pj)

            # MOTP
            match_cnt += 1
            tot_iou   += 1 - cost[gi, pj]

            # ID Switch
            if gt_id in last_match and last_match[gt_id] != trk_id:
                id_sw += 1
            last_match[gt_id] = trk_id

        miss += len(gts)   - len(matched_gt)
        fp   += len(preds) - len(matched_pr)

    mota = 1 - (miss + fp + id_sw) / tot_gt if tot_gt else 0.
    motp =  tot_iou / match_cnt          if match_cnt else 0.
    return mota, motp, id_sw

