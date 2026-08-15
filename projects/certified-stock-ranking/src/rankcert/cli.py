from __future__ import annotations
import argparse
from pathlib import Path
from .backtest import run_episodic_backtest
from .reporting import write_synthetic_report
from .synthetic import generate_synthetic_panel

def synthetic_main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=Path("results/synthetic")); p.add_argument("--months",type=int,default=240); p.add_argument("--assets",type=int,default=250); p.add_argument("--seed",type=int,default=20260814); a=p.parse_args()
    panel=generate_synthetic_panel(n_months=a.months,n_assets=a.assets,seed=a.seed)
    result=run_episodic_backtest(panel,alpha=.30,critical_value=3.2,seed=a.seed)
    write_synthetic_report(panel,result,a.output)
