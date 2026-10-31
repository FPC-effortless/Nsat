from pathlib import Path
import sys,copy,types,numpy as np,pytest
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(REPO/'dpsl_python')]


def test_nsat_source_identity_and_forecast_target_isolation(tmp_path):
 import pandas as pd,hashlib
 from ngsatml.forecast_audit import verify_source_receipts,forecast_baselines
 p=tmp_path/'raw.csv';p.write_text('fixture')
 row=verify_source_receipts([{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'publisher_url':'https://example.com/fixture'}])[0]
 assert row['identity_matches_receipt'] and row['authenticity']=='UNVERIFIED'
 dates=pd.date_range('2018-01-01',periods=50,freq='MS');f=pd.DataFrame({'month':dates,'target_month':dates+pd.offsets.MonthBegin(),'x':np.arange(50.),'price_ngn':np.arange(50.)+1,'target_price_ngn_1m':2*np.arange(50.)+1,'split_national':['train']*20+['gap']*2+['validation']*10+['gap']*2+['test']*16})
 a=forecast_baselines(f,['x']);f.loc[f.split_national=='test','target_price_ngn_1m']+=100000
 b=forecast_baselines(f,['x']);assert a['coefficients']==b['coefficients'] and a['predictions']==b['predictions']
 with pytest.raises(ValueError):forecast_baselines(f,['target_price_ngn_1m'])
