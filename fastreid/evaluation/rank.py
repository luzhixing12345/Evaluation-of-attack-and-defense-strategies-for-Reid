# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py

import warnings
from collections import defaultdict

import numpy as np

try:
    from .rank_cylib.rank_cy import evaluate_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython rank evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10

    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
                format(num_g)
        )

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    result_order = []
    q_pid_save = []
    g_pids_save = []

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        
        if q_idx < max_rank:
            result_order.append(indices[q_idx][keep][:max_rank])
            q_pid_save.append(q_pids[q_idx])
            g_pids_save.append(g_pids[result_order])
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, all_AP,0,result_order,q_pid_save,g_pids_save


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    #print("distmat.shape = ",distmat.shape)
    #print('q_pids = ',q_pids.shape)#3368
    #print('g_pids = ',g_pids.shape)#15913
    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1) #排序,替换为对应的下标值
    #print('indices = ',indices.shape)#3368x15913
    '''
    x=np.array([[1,4,3],
                [-1,6,9]])
    y=np.argsort(x,axis=1)
    [[0 2 1]
    [0 1 2]]

    x=np.array([1,4,3,-1,6,9])
    y=np.argsort(x,axis=0)

    [3 0 2 1 4 5]
    '''

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)#3368x15913
    #print('matches = ',matches.shape)
    #np.newaxis ->  create a new dimension

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    result_order = []
    q_pid_save = []
    g_pids_save = []

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]#15913
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)#15913
        #print('g_pids[order] == q_pid :',g_pids[order] == q_pid)
        #print('remove = ',remove.shape)
        
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches

        if q_idx < max_rank:
            result_order.append(indices[q_idx][keep][:max_rank].tolist())
            q_pid_save.append(q_pids[q_idx])
            g_pids_save.append(g_pids[result_order[q_idx]].tolist())

        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()# 累加 [1,1,0,1]  -> [1,2,2,3]   
        #print('cmc = ',cmc.shape)
        # <15913  已经排除了相同的目标
        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        #print('cmc_maxrank =',cmc[:max_rank])

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


def evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat,q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


def evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=50,
        use_metric_cuhk03=False,
        use_cython=True
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:

        return evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
    else:
        return evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
