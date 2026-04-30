"""Extended interpolation + hyperplane landscape analysis.

Addresses committee feedback on the interpolation_curves figures:
  1. Extend alpha outside [0, 1] to see the landscape around the chosen params.
  2. Vertical lines at the chosen params (alpha=0, alpha=1).
  3. Annotate hyperparameters per figure.
  4. Share y-axis limits between the two method pairs within each dataset.
  5. 2D hyperplane landscape across theta_DPO (= BPR-only), theta_PG, theta_SPO+.

Helpers copied verbatim from WeightInterpolationAnalysis.ipynb to keep this
script self-contained. Existing frozen outputs are not overwritten; new files
use the _ext / landscape suffix.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torch_models import PyEPONegativeBinomialRegressionModel, deterministic_bpr


# ---- Config ----

DATASET_CONFIGS = {
    'cook': {
        'data_dir': '/cluster/tufts/hugheslab/datasets/NSF_OD/cleaned/cook',
        'K': 100,
        'bird': False,
        'title': 'Cook County IL Fatal Overdoses',
        'bpr_only_path': '/cluster/tufts/hugheslab/kheuto01/opioid_hpc_test_long_big/cook/K100_bw30_nw0_ss0.01_nss100_nps100_seed123_sig0.01/best_model.pth',
        'pg_path': '/cluster/tufts/hugheslab/kheuto01/pyepo_exps/20250330_run/cook_pg_step2e-02_noise1e-02/best_model.pth',
        'spo_path': '/cluster/tufts/hugheslab/kheuto01/pyepo_exps/20250330_run/cook_spo+_step1e-01/best_model.pth',
        'pg_sigma_used': 0.01,
        'bpr_hparams': 'seed=123, bw=30, ss=0.01, n_samp=100, sig=0.01',
        'pg_hparams': 'step=2e-2, noise=1e-2',
        'spo_hparams': 'step=1e-1',
    },
    'MA': {
        'data_dir': '/cluster/tufts/hugheslab/datasets/NSF_OD/cleaned/long/MA',
        'K': 100,
        'bird': False,
        'title': 'MA Fatal Overdoses',
        'bpr_only_path': '/cluster/tufts/hugheslab/kheuto01/opioid_hpc_test_long_big/MA/K100_bw30_nw0_ss0.001_nss100_nps100_seed123_sig0.1/best_model.pth',
        'pg_path': '/cluster/tufts/hugheslab/kheuto01/pyepo_exps/20250330_run/MA_pg_step1e-02_noise8e-02/best_model.pth',
        'spo_path': '/cluster/tufts/hugheslab/kheuto01/pyepo_exps/20250330_run/MA_spo+_step2e-03/best_model.pth',
        'pg_sigma_used': 0.08,
        'bpr_hparams': 'seed=123, bw=30, ss=0.001, n_samp=100, sig=0.1',
        'pg_hparams': 'step=1e-2, noise=8e-2',
        'spo_hparams': 'step=2e-3',
    },
    'asurv': {
        'data_dir': '/cluster/tufts/hugheslab/fmuenc01/code/prob_diff_topk/data_dir/asurv/2monthly_ctxtSize5_small_5yrTrain/',
        'K': 50,
        'bird': True,
        'title': 'ANWR TX Cranes',
        'bpr_only_path': '/cluster/tufts/hugheslab/kheuto01/new_init_bird/asurv/K50_bw30_nw0_ss0.001_nss100_nps100_seed123_sig0.001/best_model.pth',
        'pg_path': '/cluster/tufts/hugheslab/kheuto01/pyepo_exps/20250414_run/asurv_pg_step1e-01_noise1e+00/best_model.pth',
        'spo_path': '/cluster/tufts/hugheslab/kheuto01/pyepo_exps/20250414_run/asurv_spo+_step8e-03/best_model.pth',
        'pg_sigma_used': 1.0,
        'bpr_hparams': 'seed=123, bw=30, ss=0.001, n_samp=100, sig=0.001',
        'pg_hparams': 'step=1e-1, noise=1e+0',
        'spo_hparams': 'step=8e-3',
    },
}

NUM_RATIO_SAMPLES = 100
PG_SIGMAS = [0.001, 0.01, 0.08, 0.1, 0.5, 1.0, 5.0]
PG_PERTURBATION_SAMPLES = 100

ALPHA_LO, ALPHA_HI, NUM_ALPHAS = -0.5, 1.5, 81
HYPERPLANE_LO, HYPERPLANE_HI, HYPERPLANE_N = -0.3, 1.3, 21


# ---- Data loading (copied from notebook) ----

def load_data(data_dir, bird=False):
    prefix = 'bird_' if bird else ''
    train_X_df = pd.read_csv(os.path.join(data_dir, f'{prefix}train_x.csv'), index_col=[0, 1])
    train_Y_df = pd.read_csv(os.path.join(data_dir, f'{prefix}train_y.csv'), index_col=[0, 1])
    val_X_df = pd.read_csv(os.path.join(data_dir, f'{prefix}valid_x.csv'), index_col=[0, 1])
    val_Y_df = pd.read_csv(os.path.join(data_dir, f'{prefix}valid_y.csv'), index_col=[0, 1])
    test_X_df = pd.read_csv(os.path.join(data_dir, f'{prefix}test_x.csv'), index_col=[0, 1])
    test_Y_df = pd.read_csv(os.path.join(data_dir, f'{prefix}test_y.csv'), index_col=[0, 1])

    def convert_df_to_3d_array(df):
        geoids = sorted(df.index.get_level_values('geoid').unique())
        timesteps = sorted(df.index.get_level_values('timestep').unique())
        geoid_to_idx = {geoid: idx for idx, geoid in enumerate(geoids)}
        X = np.zeros((len(timesteps), len(geoids), len(df.columns)))
        for (geoid, timestep), row in df.iterrows():
            X[timesteps.index(timestep), geoid_to_idx[geoid], :] = row.values
        return X, geoids, timesteps

    def convert_y_df_to_2d_array(y_df, geoids, timesteps):
        geoid_to_idx = {geoid: idx for idx, geoid in enumerate(geoids)}
        y = np.zeros((len(timesteps), len(geoids)))
        for (geoid, timestep), value in y_df.iloc[:, 0].items():
            y[timesteps.index(timestep), geoid_to_idx[geoid]] = value
        return y

    train_X, geoids, timesteps = convert_df_to_3d_array(train_X_df)
    train_time = np.array([timesteps] * len(geoids)).T
    train_y = convert_y_df_to_2d_array(train_Y_df, geoids, timesteps)
    val_X, vg, vt = convert_df_to_3d_array(val_X_df)
    val_time = np.array([vt] * len(vg)).T
    val_y = convert_y_df_to_2d_array(val_Y_df, vg, vt)
    test_X, tg, tt = convert_df_to_3d_array(test_X_df)
    test_time = np.array([tt] * len(tg)).T
    test_y = convert_y_df_to_2d_array(test_Y_df, tg, tt)

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32)

    return {
        'train': (to_t(train_X), to_t(train_time), to_t(train_y)),
        'val': (to_t(val_X), to_t(val_time), to_t(val_y)),
        'test': (to_t(test_X), to_t(test_time), to_t(test_y)),
    }


# ---- Model + interpolation helpers ----

def load_model(model_path, num_locations, num_fixed_effects, device='cuda'):
    model = PyEPONegativeBinomialRegressionModel(
        num_locations=num_locations, num_fixed_effects=num_fixed_effects,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def interpolate_params(params_a, params_b, alpha):
    return (1.0 - alpha) * params_a + alpha * params_b


@torch.no_grad()
def compute_ratio_rating(model, params_flat, X_features, time, num_samples=100):
    dist = model.build_from_single_tensor(params_flat, X_features, time)
    y_sample_MBD = dist.sample((num_samples,))
    y_sample_BMD = y_sample_MBD.permute(1, 0, 2)
    denom = y_sample_BMD.sum(dim=-1, keepdim=True) + 1.0
    ratio_rating_BMD = y_sample_BMD / denom
    return ratio_rating_BMD.mean(dim=1)


@torch.no_grad()
def compute_nll(model, params_flat, X_combined, y):
    original_params = model.params_to_single_tensor().clone()
    model.update_params(params_flat)
    nll = -model.log_likelihood(y, X_combined).detach().cpu().item()
    model.update_params(original_params)
    return nll


@torch.no_grad()
def compute_pg_loss(ratio_rating, y_true, K, sigma, num_perturbation_samples=100):
    true_topk_vals, _ = torch.topk(y_true, K, dim=-1)
    optimal_value = true_topk_vals.sum(dim=-1)
    total_regret = 0.0
    for _ in range(num_perturbation_samples):
        noise = torch.randn_like(ratio_rating) * sigma
        _, pred_idx = torch.topk(ratio_rating + noise, K, dim=-1)
        achieved = torch.gather(y_true, 1, pred_idx).sum(dim=-1)
        total_regret += ((optimal_value - achieved) / optimal_value).mean()
    return (total_regret / num_perturbation_samples).cpu().item()


@torch.no_grad()
def compute_spo_plus_loss(ratio_rating, y_true, K):
    _, true_topk_idx = torch.topk(y_true, K, dim=-1)
    pred_at_true = torch.gather(ratio_rating, 1, true_topk_idx).sum(dim=-1)
    spoofed = 2.0 * ratio_rating - y_true
    spoofed_topk, _ = torch.topk(spoofed, K, dim=-1)
    return (spoofed_topk.sum(dim=-1) - pred_at_true).mean().cpu().item()


def _compute_all_metrics(model, params_interp, val_data, test_data, K,
                        pg_sigmas, num_ratio_samples, num_pg_perturbation_samples):
    val_X, val_time, val_y = val_data
    test_X, test_time, test_y = test_data
    val_Xc = torch.cat([val_X, val_time.unsqueeze(-1)], dim=-1)
    test_Xc = torch.cat([test_X, test_time.unsqueeze(-1)], dim=-1)

    val_rr = compute_ratio_rating(model, params_interp, val_X, val_time, num_ratio_samples)
    test_rr = compute_ratio_rating(model, params_interp, test_X, test_time, num_ratio_samples)

    row = {
        'val_bpr': deterministic_bpr(val_rr, val_y, K=K).mean().cpu().item(),
        'test_bpr': deterministic_bpr(test_rr, test_y, K=K).mean().cpu().item(),
        'val_nll': compute_nll(model, params_interp, val_Xc, val_y),
        'test_nll': compute_nll(model, params_interp, test_Xc, test_y),
        'val_spo_plus': compute_spo_plus_loss(val_rr, val_y, K),
        'test_spo_plus': compute_spo_plus_loss(test_rr, test_y, K),
    }
    for sigma in pg_sigmas:
        row[f'val_pg_sigma_{sigma}'] = compute_pg_loss(val_rr, val_y, K, sigma, num_pg_perturbation_samples)
        row[f'test_pg_sigma_{sigma}'] = compute_pg_loss(test_rr, test_y, K, sigma, num_pg_perturbation_samples)
    return row


# ---- 1D sweep ----

def run_interpolation_sweep(model, params_bpr, params_other, val_data, test_data, K,
                            alphas, pg_sigmas, num_ratio_samples=100,
                            num_pg_perturbation_samples=100):
    records = []
    for i, alpha in enumerate(alphas):
        if i % 10 == 0:
            print(f'  alpha={alpha:+.2f} ({i + 1}/{len(alphas)})')
        params_interp = interpolate_params(params_bpr, params_other, alpha)
        row = _compute_all_metrics(model, params_interp, val_data, test_data, K, pg_sigmas,
                                   num_ratio_samples, num_pg_perturbation_samples)
        row['alpha'] = alpha
        records.append(row)
    cols = ['alpha'] + [c for c in records[0].keys() if c != 'alpha']
    return pd.DataFrame(records)[cols]


# ---- 2D hyperplane sweep ----

def run_hyperplane_sweep(model, params_bpr, params_pg, params_spo, val_data, test_data, K,
                         s_vals, t_vals, pg_sigmas,
                         num_ratio_samples=100, num_pg_perturbation_samples=100):
    records = []
    total = len(s_vals) * len(t_vals)
    count = 0
    for s in s_vals:
        for t in t_vals:
            count += 1
            if count % 25 == 0 or count == 1:
                print(f'  (s={s:+.2f}, t={t:+.2f})  {count}/{total}')
            params_interp = params_bpr + s * (params_pg - params_bpr) + t * (params_spo - params_bpr)
            row = _compute_all_metrics(model, params_interp, val_data, test_data, K, pg_sigmas,
                                       num_ratio_samples, num_pg_perturbation_samples)
            row['s'] = s
            row['t'] = t
            records.append(row)
    cols = ['s', 't'] + [c for c in records[0].keys() if c not in ('s', 't')]
    return pd.DataFrame(records)[cols]


# ---- Plots ----

def _hparam_text(cfg):
    return (f'K={cfg["K"]}   |   '
            f'BPR-only: {cfg["bpr_hparams"]}\n'
            f'PG: {cfg["pg_hparams"]} (sigma_train={cfg["pg_sigma_used"]})   |   '
            f'SPO+: {cfg["spo_hparams"]}')


def _parse_kv(s):
    out = {}
    for pair in s.split(','):
        k, _, v = pair.strip().partition('=')
        if k:
            out[k.strip()] = v.strip()
    return out


def _hyperplane_caption(cfg):
    bpr = _parse_kv(cfg['bpr_hparams'])
    pg = _parse_kv(cfg['pg_hparams'])
    spo = _parse_kv(cfg['spo_hparams'])
    return (f'K={cfg["K"]}   |   '
            f'BPR: $\\sigma$={bpr.get("sig", "?")}, step size={bpr.get("ss", "?")}   |   '
            f'PG: $h$={pg.get("noise", "?")}, step size={pg.get("step", "?")}   |   '
            f'SPO+: step size={spo.get("step", "?")}')


def plot_interpolation_curves_ext(df, dataset_name, method_pair, cfg, pg_sigmas, ylims=None,
                                  save=True, suffix='_ext'):
    """Extended-range 2x2 interpolation plot.

    ylims: dict {metric: (lo, hi)} for within-dataset sharing across the two method pairs.
    """
    other_name = method_pair.split('_to_')[1]
    if other_name == 'SPO':
        other_name = 'SPO+'
    K = cfg['K']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    fig.suptitle(
        f'{cfg["title"]} (K={K})\n'
        f'Interpolation: $\\theta_\\alpha = (1-\\alpha)\\theta_{{DPO}} + \\alpha\\theta_{{{other_name}}}$',
        fontsize=15,
    )

    alpha = df['alpha'].values

    def _decorate(ax, metric_key, ylabel, title):
        # Shade extrapolated regions
        ax.axvspan(alpha.min(), 0.0, color='grey', alpha=0.08, zorder=0)
        ax.axvspan(1.0, alpha.max(), color='grey', alpha=0.08, zorder=0)
        # Vertical lines at chosen parameters
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1.3, alpha=0.75,
                   label=r'chosen $\theta_{DPO}$ ($\alpha$=0)')
        ax.axvline(1.0, color='k', linestyle=':', linewidth=1.3, alpha=0.75,
                   label=rf'chosen $\theta_{{{other_name}}}$ ($\alpha$=1)')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ylims is not None and metric_key in ylims:
            ax.set_ylim(ylims[metric_key])

    # Top-left: BPR
    ax = axes[0, 0]
    ax.plot(alpha, df['val_bpr'], 'b-o', markersize=2, label='Validation', alpha=0.8)
    ax.plot(alpha, df['test_bpr'], 'r-s', markersize=2, label='Test', alpha=0.8)
    _decorate(ax, 'bpr', 'BPR (higher = better)', 'Best Possible Ratio (BPR)')
    ax.legend(fontsize=8, loc='best')

    # Top-right: NLL
    ax = axes[0, 1]
    ax.plot(alpha, df['val_nll'], 'b-o', markersize=2, label='Validation', alpha=0.8)
    ax.plot(alpha, df['test_nll'], 'r-s', markersize=2, label='Test', alpha=0.8)
    _decorate(ax, 'nll', 'NLL (lower = better fit)', 'Negative Log-Likelihood')
    ax.legend(fontsize=8, loc='best')

    # Bottom-left: SPO+
    ax = axes[1, 0]
    ax.plot(alpha, df['val_spo_plus'], 'b-o', markersize=2, label='Validation', alpha=0.8)
    ax.plot(alpha, df['test_spo_plus'], 'r-s', markersize=2, label='Test', alpha=0.8)
    _decorate(ax, 'spo_plus', 'SPO+ Loss (lower = better)', 'SPO+ Surrogate Loss')
    ax.legend(fontsize=8, loc='best')

    # Bottom-right: PG at multiple sigmas + empirical regret overlay
    ax = axes[1, 1]
    cmap = cm.viridis
    pg_used = cfg.get('pg_sigma_used')
    for i, sigma in enumerate(pg_sigmas):
        color = cmap(i / max(len(pg_sigmas) - 1, 1))
        is_used = pg_used is not None and sigma == pg_used
        ax.plot(alpha, df[f'test_pg_sigma_{sigma}'], color=color,
                linewidth=2.5 if is_used else 1.0,
                linestyle='-' if is_used else '--',
                label=f'$\\sigma$={sigma}' + (' (training)' if is_used else ''),
                alpha=0.9)
    # Empirical regret = 1 - test_bpr
    ax.plot(alpha, 1.0 - df['test_bpr'].values, color='black', linewidth=1.6,
            linestyle='-', label='Actual regret (1 - test BPR)', alpha=0.9)
    _decorate(ax, 'pg', 'PG surrogate / regret (lower = better)',
              'PG Surrogate Loss (Test)')
    ax.legend(fontsize=7, loc='best')
    ax.text(0.02, -0.22,
            r'Large $\sigma$: smoothed surrogate diverges from true regret '
            r'(expected for finite-difference smoothing).',
            transform=ax.transAxes, fontsize=8, style='italic', color='dimgray')

    # Hyperparam footer
    fig.text(0.5, 0.005, _hparam_text(cfg), ha='center', fontsize=8, color='#333')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save:
        fname = f'interpolation_curves_{dataset_name}_{method_pair}{suffix}.pdf'
        plt.savefig(fname, dpi=600, bbox_inches='tight')
        print(f'  saved {fname}')
    plt.close(fig)


def plot_hyperplane_landscape(df_hyper, dataset_name, cfg, pg_sigmas, save=True):
    """2D hyperplane landscape: one figure per dataset, one heatmap per metric."""
    K = cfg['K']
    pg_used = cfg['pg_sigma_used']
    # Use the nearest PG sigma that was actually swept (pg_used may not be in the sweep list)
    pg_nearest = min(pg_sigmas, key=lambda s: abs(s - pg_used))
    pg_label = f'$\\sigma$={pg_nearest}'
    if abs(pg_nearest - pg_used) > 1e-9:
        pg_label += f' (nearest to training $\\sigma$={pg_used})'
    metrics = [
        ('test_bpr', 'Test BPR (higher = better)', 'viridis', False),
        ('test_nll', 'Test NLL (lower = better)', 'viridis_r', True),
        ('test_spo_plus', 'Test SPO+ loss (lower = better)', 'viridis_r', False),
        (f'test_pg_sigma_{pg_nearest}',
         f'Test PG loss @ {pg_label}',
         'viridis_r', False),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.3 * len(metrics), 5.2))
    fig.suptitle(
        f'{cfg["title"]} (K={K})\n'
        r'$\theta(s,t) = (1-s-t)\,\theta_{DPO} + s\,\theta_{PG} + t\,\theta_{SPO+}$',
        fontsize=14,
    )

    anchors = {
        r'$\theta_{DPO}$': (0.0, 0.0),
        r'$\theta_{PG}$': (1.0, 0.0),
        r'$\theta_{SPO+}$': (0.0, 1.0),
    }

    s_vals = np.sort(df_hyper['s'].unique())
    t_vals = np.sort(df_hyper['t'].unique())
    ds_s = s_vals[1] - s_vals[0]
    ds_t = t_vals[1] - t_vals[0]
    extent = [s_vals.min() - ds_s / 2, s_vals.max() + ds_s / 2,
              t_vals.min() - ds_t / 2, t_vals.max() + ds_t / 2]

    for ax, (metric, label, cmap_name, log_norm) in zip(axes, metrics):
        pivot = df_hyper.pivot(index='t', columns='s', values=metric)
        pivot = pivot.loc[t_vals, s_vals]
        Z = pivot.values

        # NLL gets log-scale normalization (often large range)
        if log_norm:
            import matplotlib.colors as mcolors
            finite = np.isfinite(Z) & (Z > 0)
            vmin = np.nanpercentile(Z[finite], 5) if finite.any() else 1.0
            vmax = np.nanpercentile(Z[finite], 95) if finite.any() else vmin * 10
            if vmax > vmin:
                norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
            else:
                norm = None
        else:
            norm = None

        im = ax.imshow(Z, aspect='auto', origin='lower', extent=extent,
                       cmap=cmap_name, norm=norm)
        # Contours
        try:
            cs = ax.contour(s_vals, t_vals, Z, levels=8, colors='white',
                            linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.3g')
        except Exception:
            pass

        # Anchor markers
        for name, (s, t) in anchors.items():
            ax.plot(s, t, marker='*', markersize=18, markerfacecolor='white',
                    markeredgecolor='black', markeredgewidth=1.3, zorder=5)
            ax.annotate(name, (s, t), textcoords='offset points', xytext=(8, 8),
                        fontsize=11, color='black',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec='black', alpha=0.7))

        ax.set_xlabel(r'$s$ (along $\theta_{PG} - \theta_{DPO}$)')
        ax.set_ylabel(r'$t$ (along $\theta_{SPO+} - \theta_{DPO}$)')
        ax.set_title(label, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.text(0.5, 0.005, _hyperplane_caption(cfg), ha='center', fontsize=8, color='#333')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    if save:
        fname = f'hyperplane_landscape_{dataset_name}.pdf'
        plt.savefig(fname, dpi=600, bbox_inches='tight')
        print(f'  saved {fname}')
    plt.close(fig)


# ---- Drivers ----

def _ylims_from_two(df_a, df_b, pg_sigmas):
    def lim(*arrays):
        a = np.concatenate([np.asarray(x) for x in arrays])
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        lo, hi = a.min(), a.max()
        pad = 0.05 * (hi - lo if hi > lo else abs(hi) + 1e-6)
        return (lo - pad, hi + pad)

    out = {}
    out['bpr'] = lim(df_a['val_bpr'], df_a['test_bpr'], df_b['val_bpr'], df_b['test_bpr'])
    out['nll'] = lim(df_a['val_nll'], df_a['test_nll'], df_b['val_nll'], df_b['test_nll'])
    out['spo_plus'] = lim(df_a['val_spo_plus'], df_a['test_spo_plus'],
                          df_b['val_spo_plus'], df_b['test_spo_plus'])
    # PG ylim covers all sigmas' test curves + empirical regret
    pg_cols = [f'test_pg_sigma_{s}' for s in pg_sigmas]
    series = [df_a[c] for c in pg_cols] + [df_b[c] for c in pg_cols]
    series += [1.0 - df_a['test_bpr'], 1.0 - df_b['test_bpr']]
    out['pg'] = lim(*series)
    return out


def run_1d_sweeps(datasets, device):
    alphas = np.linspace(ALPHA_LO, ALPHA_HI, NUM_ALPHAS)
    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        print(f'\n{"=" * 60}\n[1D] Dataset: {ds_name} (K={cfg["K"]})\n{"=" * 60}')
        data = load_data(cfg['data_dir'], bird=cfg['bird'])
        val_data = tuple(x.to(device) for x in data['val'])
        test_data = tuple(x.to(device) for x in data['test'])
        num_loc = data['train'][0].shape[1]
        num_fe = data['train'][0].shape[2]

        model_bpr = load_model(cfg['bpr_only_path'], num_loc, num_fe, device)
        params_bpr = model_bpr.params_to_single_tensor().detach().clone()
        model_pg = load_model(cfg['pg_path'], num_loc, num_fe, device)
        params_pg = model_pg.params_to_single_tensor().detach().clone()
        model_spo = load_model(cfg['spo_path'], num_loc, num_fe, device)
        params_spo = model_spo.params_to_single_tensor().detach().clone()

        print(f'  BPR->PG sweep')
        df_pg = run_interpolation_sweep(
            model_bpr, params_bpr, params_pg, val_data, test_data, cfg['K'],
            alphas=alphas, pg_sigmas=PG_SIGMAS,
            num_ratio_samples=NUM_RATIO_SAMPLES,
            num_pg_perturbation_samples=PG_PERTURBATION_SAMPLES,
        )
        df_pg.to_csv(f'interpolation_{ds_name}_BPR_to_PG_ext.csv', index=False)

        print(f'  BPR->SPO+ sweep')
        df_spo = run_interpolation_sweep(
            model_bpr, params_bpr, params_spo, val_data, test_data, cfg['K'],
            alphas=alphas, pg_sigmas=PG_SIGMAS,
            num_ratio_samples=NUM_RATIO_SAMPLES,
            num_pg_perturbation_samples=PG_PERTURBATION_SAMPLES,
        )
        df_spo.to_csv(f'interpolation_{ds_name}_BPR_to_SPO_ext.csv', index=False)


def run_hyperplane(datasets, device):
    s_vals = np.linspace(HYPERPLANE_LO, HYPERPLANE_HI, HYPERPLANE_N)
    t_vals = np.linspace(HYPERPLANE_LO, HYPERPLANE_HI, HYPERPLANE_N)
    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        print(f'\n{"=" * 60}\n[HP] Dataset: {ds_name} (K={cfg["K"]})\n{"=" * 60}')
        data = load_data(cfg['data_dir'], bird=cfg['bird'])
        val_data = tuple(x.to(device) for x in data['val'])
        test_data = tuple(x.to(device) for x in data['test'])
        num_loc = data['train'][0].shape[1]
        num_fe = data['train'][0].shape[2]

        model_bpr = load_model(cfg['bpr_only_path'], num_loc, num_fe, device)
        params_bpr = model_bpr.params_to_single_tensor().detach().clone()
        params_pg = load_model(cfg['pg_path'], num_loc, num_fe, device).params_to_single_tensor().detach().clone()
        params_spo = load_model(cfg['spo_path'], num_loc, num_fe, device).params_to_single_tensor().detach().clone()

        df_hp = run_hyperplane_sweep(
            model_bpr, params_bpr, params_pg, params_spo,
            val_data, test_data, cfg['K'],
            s_vals=s_vals, t_vals=t_vals, pg_sigmas=PG_SIGMAS,
            num_ratio_samples=NUM_RATIO_SAMPLES,
            num_pg_perturbation_samples=PG_PERTURBATION_SAMPLES,
        )
        df_hp.to_csv(f'hyperplane_{ds_name}.csv', index=False)


def make_plots(datasets):
    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        df_pg_path = Path(f'interpolation_{ds_name}_BPR_to_PG_ext.csv')
        df_spo_path = Path(f'interpolation_{ds_name}_BPR_to_SPO_ext.csv')
        if not (df_pg_path.exists() and df_spo_path.exists()):
            print(f'  [plots] skipping {ds_name}: missing ext CSV')
            continue
        df_pg = pd.read_csv(df_pg_path)
        df_spo = pd.read_csv(df_spo_path)
        ylims = _ylims_from_two(df_pg, df_spo, PG_SIGMAS)
        print(f'\n[plots] {ds_name}: ylims={ylims}')
        plot_interpolation_curves_ext(df_pg, ds_name, 'BPR_to_PG', cfg, PG_SIGMAS, ylims=ylims)
        plot_interpolation_curves_ext(df_spo, ds_name, 'BPR_to_SPO', cfg, PG_SIGMAS, ylims=ylims)

        hp_path = Path(f'hyperplane_{ds_name}.csv')
        if hp_path.exists():
            df_hp = pd.read_csv(hp_path)
            plot_hyperplane_landscape(df_hp, ds_name, cfg, PG_SIGMAS)
        else:
            print(f'  [plots] no hyperplane CSV for {ds_name}')


def make_hyperplane_plots(datasets):
    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        hp_path = Path(f'hyperplane_{ds_name}.csv')
        if not hp_path.exists():
            print(f'  [hp_plots] skipping {ds_name}: no hyperplane CSV')
            continue
        df_hp = pd.read_csv(hp_path)
        plot_hyperplane_landscape(df_hp, ds_name, cfg, PG_SIGMAS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['1d', 'hp', 'plots', 'hp_plots', 'all'], default='all')
    ap.add_argument('--datasets', nargs='*', default=list(DATASET_CONFIGS.keys()))
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  stage={args.stage}  datasets={args.datasets}')

    if args.stage in ('1d', 'all'):
        run_1d_sweeps(args.datasets, device)
    if args.stage in ('hp', 'all'):
        run_hyperplane(args.datasets, device)
    if args.stage in ('plots', 'all'):
        make_plots(args.datasets)
    if args.stage == 'hp_plots':
        make_hyperplane_plots(args.datasets)


if __name__ == '__main__':
    main()
