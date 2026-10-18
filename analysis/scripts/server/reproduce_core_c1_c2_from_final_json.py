#!/usr/bin/env python3
"""Independent, reconstructed Core Base C1/C2 reproducibility pipeline.

NOT the historical original plotting script. Never writes into publication folder.
Requires numpy, pandas, scipy, matplotlib. All 30 final Core Base JSONs required.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STRATEGIES = ['fedavg','heuristic','dqn','ddqn','tl_ppo','tr_dp_afrl']
LABELS = dict(zip(STRATEGIES,['FedAvg','Heuristic','DQN','DDQN','TL-PPO','TR-DP-AFRL']))
SEEDS = [42,43,44,45,46]
METRICS = ['acc','reward','latency_s','train_time_s','offload_intensity','churn_ratio','selected_clients']
PAIR_METRICS = ['acc','reward','latency_s','train_time_s']


def load(root):
    all_runs = {}
    for strategy in STRATEGIES:
        for seed in SEEDS:
            path = root / f'results_{strategy}_CORE_FINAL_seed{seed}.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            d = json.loads(path.read_text())
            for field in ['round','acc','reward','latency_s','train_time_s','offload_intensity','churn_ratio','selected_clients_per_round']:
                if field not in d or len(d[field]) != 300:
                    raise ValueError(f'{path.name}: invalid {field}')
            if list(d['round']) != list(range(1,301)):
                raise ValueError(f'{path.name}: rounds not 1..300')
            counts = [len(v) if isinstance(v,list) else int(v) for v in d['selected_clients_per_round']]
            expected = 5 if strategy == 'tr_dp_afrl' else 10
            if any(v != expected for v in counts):
                raise ValueError(f'{path.name}: participation count differs from {expected}')
            d['_selected_count'] = counts
            all_runs[(strategy,seed)] = d
    return all_runs


def tables(runs):
    seedrows = []
    for strategy in STRATEGIES:
        for seed in SEEDS:
            d = runs[(strategy,seed)]
            row = dict(strategy=strategy,seed=seed,steady_state_start=251,steady_state_end=300,n_rounds=50)
            for m in METRICS:
                values = d['_selected_count'] if m == 'selected_clients' else d[m]
                row[m] = float(np.mean(np.asarray(values[250:300],dtype=float)))
            seedrows.append(row)
    seed_df = pd.DataFrame(seedrows)
    summaryrows=[]
    for strategy in STRATEGIES:
        part=seed_df[seed_df.strategy==strategy]
        row=dict(strategy=strategy,n_seeds=5)
        for m in METRICS:
            row[m+'_mean']=float(part[m].mean())
            row[m+'_sd']=float(part[m].std(ddof=1))
        summaryrows.append(row)
    summary_df=pd.DataFrame(summaryrows)
    pairs=[]
    for metric in PAIR_METRICS:
        for baseline in STRATEGIES:
            if baseline=='tr_dp_afrl': continue
            a=seed_df[seed_df.strategy=='tr_dp_afrl'].set_index('seed').loc[SEEDS,metric].to_numpy()
            b=seed_df[seed_df.strategy==baseline].set_index('seed').loc[SEEDS,metric].to_numpy()
            delta=a-b
            sd=float(np.std(delta,ddof=1))
            w=wilcoxon(delta,zero_method='wilcox',correction=False,alternative='two-sided',method='exact')
            pairs.append(dict(metric=metric,comparison=f'tr_dp_afrl_vs_{baseline}',n_pairs=5,
                              mean_difference=float(np.mean(delta)),median_difference=float(np.median(delta)),
                              sd_difference=sd,cohen_dz=float(np.mean(delta)/sd) if sd else float('nan'),
                              tr_higher_count=int(np.sum(delta>0)),tr_lower_count=int(np.sum(delta<0)),
                              wilcoxon_W=float(w.statistic),wilcoxon_exact_p_two_sided=float(w.pvalue)))
    return {'CORE_C1_SEED_LEVEL_STEADY_STATE.csv':seed_df,
            'CORE_C1_SUMMARY.csv':summary_df,
            'CORE_C2_PAIRED_STATISTICS.csv':pd.DataFrame(pairs)}


def verify(tables, original, atol):
    failures=0
    for name,new in tables.items():
        path=original/name
        if not path.is_file():
            print('MISSING ORIGINAL:',path); failures+=1;continue
        old=pd.read_csv(path)
        if list(new.columns)!=list(old.columns) or len(new)!=len(old):
            print('SCHEMA/ROW MISMATCH:',name);failures+=1;continue
        local_fail=[]
        for col in new.columns:
            if pd.api.types.is_numeric_dtype(old[col]):
                av=pd.to_numeric(new[col]).to_numpy(float)
                bv=pd.to_numeric(old[col]).to_numpy(float)
                if not np.allclose(av,bv,atol=atol,rtol=0,equal_nan=True):
                    local_fail.append((col,float(np.nanmax(np.abs(av-bv)))))
            elif new[col].astype(str).tolist()!=old[col].astype(str).tolist():
                local_fail.append((col,'text mismatch'))
        if local_fail:
            failures+=1;print('FAIL:',name,local_fail)
        else:
            print('PASS:',name,f'({len(new)} rows, tolerance {atol:g})')
    return failures


def figures(runs, summary, out):
    # C1: independent visual reconstruction, NOT a pixel-identical historical reproduction.
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for strategy in STRATEGIES:
        for metric,ax in zip(['acc','reward','latency_s','offload_intensity'],axs.flat):
            curves=np.array([runs[(strategy,seed)][metric] for seed in SEEDS],dtype=float)
            ax.plot(range(1,301),curves.mean(axis=0),label=LABELS[strategy],linewidth=1.3)
            ax.set_title({'acc':'Accuracy','reward':'Reward','latency_s':'Round latency (s)','offload_intensity':'Offloading intensity'}[metric])
            ax.grid(alpha=.2)
            ax.set_xlabel('Communication round')
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=6,fontsize=8)
    fig.tight_layout(rect=[0,.055,1,1])
    for ext in ['png','pdf']:
        fig.savefig(out/f'CORE_C1_CONVERGENCE_2x2_RECONSTRUCTED.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

    # C2: follows the historical Bash-history plotting logic supplied by the user.
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    for _,r in summary.iterrows():
        strategy=r['strategy']; special=strategy=='tr_dp_afrl'
        ax.errorbar(r['latency_s_mean'],r['acc_mean'],xerr=r['latency_s_sd'],yerr=r['acc_sd'],
                    fmt='none',capsize=3,linewidth=1.0,alpha=.75,zorder=10 if special else 5)
        ax.scatter(r['latency_s_mean'],r['acc_mean'],marker='*' if special else 'o',
                   s=220 if special else 70,zorder=10 if special else 5)
        ax.annotate(LABELS[strategy],(r['latency_s_mean'],r['acc_mean']),
                    xytext=(7,6),textcoords='offset points',fontsize=9)
    ax.set_xlabel('Steady-state latency (s) ↓')
    ax.set_ylabel('Steady-state accuracy ↑')
    ax.set_title('Core Benchmark: Accuracy–Latency Trade-off')
    ax.grid(True,alpha=.25)
    ax.text(.02,.02,'Rounds 251–300; mean ± SD across 5 seeds',transform=ax.transAxes,fontsize=8,va='bottom')
    for ext in ['png','pdf']:
        fig.savefig(out/f'CORE_C2_ACCURACY_LATENCY_PARETO_RECONSTRUCTED.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path('/home/ubuntu/afrl_runs/CORE_FINAL_300R_5SEED'))
    p.add_argument('--publication',type=Path,default=Path('/home/ubuntu/afrl_runs/CORE_FINAL_PUBLICATION'))
    p.add_argument('--output',type=Path,default=Path('/home/ubuntu/afrl_runs/CORE_C1_C2_REPRO_CHECK'))
    p.add_argument('--tolerance',type=float,default=1e-10)
    args=p.parse_args()
    if args.output.resolve()==args.publication.resolve() or args.publication.resolve() in args.output.resolve().parents:
        p.error('Output cannot be publication folder or within it')
    runs=load(args.results)
    t=tables(runs)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,df in t.items(): df.to_csv(args.output/name,index=False)
    figures(runs,t['CORE_C1_SUMMARY.csv'],args.output)
    print('VALIDATED: 30 JSON files, 9000 rounds, expected K counts')
    failures=verify(t,args.publication,args.tolerance)
    print('REPRODUCTION OUTPUT:',args.output)
    print('NOTE: figures are independently reconstructed; image/PDF byte equality is NOT asserted.')
    if failures: raise SystemExit(f'{failures} numerical verification failure(s)')
    print('PASS: all three publication CSVs numerically reproduced')

if __name__=='__main__': main()
