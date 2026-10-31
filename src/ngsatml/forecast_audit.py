"""CAP004 source receipts and chronological baseline forecasting.

A matching digest establishes artifact identity, not publisher authenticity.
There is no synthetic-to-authentic promotion in this interface.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd

def verify_source_receipts(receipts):
    rows=[]
    for item in receipts:
        path=Path(item['path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        identity_ok=digest==item['sha256']
        url=item.get('publisher_url','');valid_url=urlparse(url).scheme=='https' and bool(urlparse(url).hostname)
        rows.append({'path':str(path),'sha256':digest,'identity_matches_receipt':identity_ok,
                     'has_https_publisher_reference':valid_url,'authenticity':'UNVERIFIED',
                     'eligible_for_source_review':identity_ok and valid_url})
    return rows

def forecast_baselines(frame,features,target='target_price_ngn_1m',split='split_national',ridge=1.):
    """Fit only train rows; never remove validation/test rows using target outliers."""
    forbidden={'quality_suspicious','target_within_split','national_target_within_split','target_split_national','target_split'}
    if ridge<=0 or not features or target in features or any('target' in f.lower() or f in forbidden for f in features):
        raise ValueError('Need positive ridge and explicit input-only features')
    required=[*features,target,split,'month','target_month']
    if any(c not in frame for c in required):raise ValueError('Missing forecast columns')
    data=frame.copy();dates=pd.to_datetime(data['month'],utc=True);targets=pd.to_datetime(data['target_month'],utc=True)
    if dates.isna().any() or targets.isna().any() or (targets<=dates).any():raise ValueError('Targets must occur strictly after valid input dates')
    if 'feature_available_at' in data and (pd.to_datetime(data['feature_available_at'],utc=True)>dates).any():raise ValueError('A feature arrives after the prediction time')
    train=data[split].eq('train');val=data[split].eq('validation');test=data[split].eq('test')
    if not train.any() or not val.any() or not test.any():raise ValueError('Need nonempty train, validation and test')
    if targets[train].max()>=dates[val|test].min() or targets[val].max()>=dates[test].min():raise ValueError('Training/evaluation horizons overlap')
    X=data[features].apply(pd.to_numeric,errors='coerce').to_numpy(float);y=pd.to_numeric(data[target],errors='coerce').to_numpy(float)
    # Train-only imputation and scaling; test labels cannot affect fitted coefficients.
    tr=train.to_numpy();mean=np.nanmean(X[tr],axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(y[tr]).all():raise ValueError('Invalid training columns/targets')
    X=np.where(np.isfinite(X),X,mean);std=X[tr].std(0);std=np.where(std>1e-9,std,1.);X=(X-mean)/std
    Z=np.column_stack([np.ones(len(X)),X]);penalty=np.diag([0.,*([ridge]*len(features))]);coef=np.linalg.solve(Z[tr].T@Z[tr]+penalty,Z[tr].T@y[tr]);pred=Z@coef
    last=pd.to_numeric(data.get('price_ngn',pd.Series(np.nan,index=data.index)),errors='coerce').to_numpy(float)
    if target in {'target_log_change_1m','target_change_1m_pct'}:
        last=np.zeros(len(data))
    rows=[]
    for label,mask in [('validation',val.to_numpy()),('test',test.to_numpy())]:
        valid=mask&np.isfinite(y);baseline=valid&np.isfinite(last)
        if valid.sum()!=mask.sum():raise ValueError('Evaluation targets must all be finite; no silent row drops')
        rows.append({'split':label,'rows':int(mask.sum()),'ridge_mae':float(np.mean(abs(pred[valid]-y[valid]))),
                     'persistence_mae':float(np.mean(abs(last[baseline]-y[baseline]))) if baseline.any() else None})
    return {'features':features,'fit_rows':int(tr.sum()),'coefficients':coef.tolist(),'feature_mean':mean.tolist(),'predictions':pred.tolist(),'scores':rows,'source_authenticity':'UNVERIFIED_BY_THIS_FUNCTION','target':target,'metric_units':'target units','availability_vintage_verified':False}
