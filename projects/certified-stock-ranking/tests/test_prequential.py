import numpy as np
import pandas as pd
from rankcert.prequential import PrequentialConfig,generate_prequential_predictions

def test_prequential_output_is_chronological():
    rng=np.random.default_rng(0); dates=pd.date_range("2000-01-31",periods=30,freq="ME"); rows=[]
    for d in dates:
        for i in range(20):
            x=rng.normal(); rows.append({"date":d,"asset_id":i,"x":x,"ret_fwd":.01*x+rng.normal(0,.05)})
    panel=pd.DataFrame(rows)
    out=generate_prequential_predictions(panel,["x"],config=PrequentialConfig(train_months=12,refit_frequency_months=6,scale_window_months=6,min_scale_observations=10_000))
    assert out.date.min()==dates[12]
    assert np.all(out.scale>0) and out.prediction.notna().all()

def test_fixed_grid_backtest_needs_no_proposal_block():
    from rankcert.backtest import run_episodic_backtest
    rng=np.random.default_rng(1); dates=pd.date_range("2000-01-31",periods=36,freq="ME"); rows=[]
    for d in dates:
        p=rng.normal(size=30); y=.1*p+rng.normal(size=30)
        for i in range(30): rows.append({"date":d,"asset_id":i,"prediction":p[i],"scale":1.0,"ret_fwd":y[i]})
    out=run_episodic_backtest(pd.DataFrame(rows),certification_months=24,deployment_months=12,proposal_months=0,candidate_grid=[0,.5,1],alpha=.5,critical_value=2.0)
    assert len(out.epochs)==1 and out.epochs.n_proposal_months.iloc[0]==0
