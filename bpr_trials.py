import numpy as np

def calc_bpr_many_trials(
        dist_N, K=3, n_trials=10000,
        s_samples=10000, seed=101,
        strategy='pick_mean',
        percentile_as_frac=0.95):
    N = len(dist_N)
    y_RN = np.zeros((n_trials, N), dtype=np.int32)


    for n, dist in enumerate(dist_N):
        random_state = np.random.RandomState(10000 * seed + n)
        y_RN[:, n] = dist.rvs(size=n_trials, random_state=random_state)

    if strategy == 'cross_ratio':
        S = s_samples*n_trials
        y_SN = np.zeros((S, N))
        sum_str_N = [None for _ in range(N)]
        for n, dist in enumerate(dist_N):
            random_state = np.random.RandomState(10000 * seed + n)
            y_SN[:,n] = dist.rvs(size=S, random_state=random_state)
            sum_str_N[n] = " ".join(['%.1f' % np.percentile(y_SN[:,n], p)
                                    for p in [0, 10, 50, 90, 100]])
        ratio_N = np.mean(y_SN / np.sum(y_SN, axis=1, keepdims=1), axis=0)
        assert ratio_N.shape == (N,)
        selected_ids_K = np.argsort(-1 * ratio_N)[:K]

        selected_ids_RK = np.tile(selected_ids_K, (n_trials,1))

    if strategy == 'cross_ratio_topk':
        S = s_samples*n_trials
        y_SN = np.zeros((S, N))
        sum_str_N = [None for _ in range(N)]
        for n, dist in enumerate(dist_N):
            random_state = np.random.RandomState(10000 * seed + n)
            y_SN[:,n] = dist.rvs(size=S, random_state=random_state)
            sum_str_N[n] = " ".join(['%.1f' % np.percentile(y_SN[:,n], p)
                                    for p in [0, 10, 50, 90, 100]])

        topk_ids_SN = np.argsort(-1 * y_SN, axis=1)[:, :K]
        topk_y_SN = np.take_along_axis(y_SN, topk_ids_SN, axis=1)
        ratiotopk_N = np.mean(y_SN / np.sum(topk_y_SN, axis=1, keepdims=1), axis=0)
        assert ratiotopk_N.shape == (N,)
        selected_ids_K = np.argsort(-1 * ratiotopk_N)[:K]

        selected_ids_RK = np.tile(selected_ids_K, (n_trials,1))

    if strategy == 'guess_random':
        random_state = np.random.RandomState(10000 * seed)
        selected_ids_RK = np.zeros((n_trials, K), dtype=np.int32)
        for trial in range(n_trials):
            selected_ids_RK[trial,:] = random_state.permutation(N)[:K]

    if strategy.count('pick'):
        score_N = np.zeros(N)
        sum_str_N = [None for _ in range(N)]
        for n, dist in enumerate(dist_N):
            random_state = np.random.RandomState(10000 * seed + n)
            y_samples_S = dist.rvs(size=s_samples*n_trials, random_state=random_state)
            sum_str_N[n] = " ".join(['%.1f' % np.percentile(y_samples_S, p)
                                    for p in [0, 10, 50, 90, 100]])
            
            if strategy == 'pick_mean':
                score_N[n] = np.mean(y_samples_S)
            if strategy == 'pick_consistent':
                score_N[n] = 1 if n < 4 else 0
            if strategy == 'pick_variance':
                score_N[n] = 1 if n >=4 and n < 8 else 0
            if strategy == 'pick_lottery':
                score_N[n] = 1 if n >=8 else 0
            if strategy == 'pick_each':
                score_N[n] = 1 if n == 0 or n==4 or n==7 else 0
            if strategy == 'pick_median':
                score_N[n] = np.median(y_samples_S)
            elif strategy == 'pick_percentile':
                score_N[n] = np.percentile(y_samples_S, percentile_as_frac)  

        selected_ids_K = np.argsort(-1 * score_N)[:K]

        selected_ids_RK = np.tile(selected_ids_K, (n_trials,1))

    yselect_RK = np.take_along_axis(y_RN, selected_ids_RK, axis=1)
    topk_ids_RK = np.argsort(-1 * y_RN, axis=1)[:, :K]
    ytop_RK = np.take_along_axis(y_RN, topk_ids_RK, axis=1)

    numer_R = np.sum(yselect_RK, axis=1)
    denom_R = np.sum(ytop_RK, axis=1)
    
    assert np.all(numer_R <= denom_R + 1e-10)
    
    return numer_R / denom_R
