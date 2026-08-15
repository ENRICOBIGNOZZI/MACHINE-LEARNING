import numpy as np
from rankcert.snrcps import certify_monotone_losses,monotone_envelope,ordered_selector

def test_monotone_envelope():
    losses=np.array([[.4,.5,.2],[.2,.1,.3]]); out=monotone_envelope(losses)
    assert np.all(np.diff(out,axis=1)<=1e-12) and np.all(out>=losses)

def test_ordered_selector():
    assert ordered_selector([.8,.4,.3],.5)==1
    assert ordered_selector([.8,.6,.7],.5) is None

def test_certification_returns_candidate():
    rng=np.random.default_rng(1); losses=np.column_stack([rng.binomial(1,p,size=120) for p in [.55,.4,.25]])
    result=certify_monotone_losses(losses,[0,.5,1],alpha=.5,delta=.1,critical_value=2.5)
    assert result.selected_index in {1,2}
