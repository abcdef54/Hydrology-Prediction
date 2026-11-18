"""Score every epoch of training runs on every split (train/validation/purged/test/new), all horizons of the run.

Usage: PYTHONPATH=. .venv/bin/python research/scripts/score_runs_all_splits.py EXT_DATASET RUN_DIR [RUN_DIR...]
EXT_DATASET: the run dataset rebuilt on a longer grid with --normalizers-from (issue times after 2026-10-14 19:00 UTC = "new").
Writes reports/scores/<dataset>__<run>.parquet (error counts, MAE, max per epoch/split/horizon) and .npz (predictions per epoch)."""
import sys, joblib, numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.ai.regional_training_data import RegionalTrainingDataset, regional_collate
from src.ai.sequence_checkpoint import SequenceRuntime
S=Path(sys.argv[0]).resolve().parent.parent.parent/'reports'; (S/'scores').mkdir(parents=True, exist_ok=True)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
EXT=sys.argv[1]
for run in map(Path,sys.argv[2:]):
    eps=sorted(p for p in run.rglob('epoch_*/model.pt') if p.parent.name!='epoch_000')
    saved=joblib.load(eps[0].with_suffix('.preprocessing.joblib'))['regional']
    kw=dict(rain_stations='+'.join(saved['rain_stations']) or 'none',use_weather_forecast=False,time_features=saved['time_features'],
            target_horizons=saved['target_horizons'],sequence_length=saved['training_selection']['sequence_length'])
    tn=None; data=[]
    for sp in ['train','validation','purged','test_exploratory']:
        ds=RegionalTrainingDataset(EXT,split=sp,target_normalizers=tn,**kw); tn=tn or ds.view.normalizers
        fr=ds.view.samples; a=pd.to_datetime(fr.anchor_utc,utc=True)
        lab=np.where(sp=='test_exploratory',np.where(a>pd.Timestamp('2026-10-14 19:00',tz='UTC'),'new','test'),sp)
        data.append((list(DataLoader(ds,batch_size=1024,collate_fn=regional_collate)),lab,np.stack([fr[f'target_h{h}_mm'] for h in saved['target_horizons']],1),a.to_numpy()))
    lab=np.concatenate([d[1] for d in data]); Y=np.concatenate([d[2] for d in data]); A=np.concatenate([d[3] for d in data])
    rows=[]; preds=[]
    for p in eps:
        rt=SequenceRuntime(p,device=DEV); ep=int(p.parent.name.split('_')[1])
        P=np.concatenate([rt.predict_mm(b['x'],b['station_ids']) for d in data for b in d[0]]); preds.append(P.astype(np.float32))
        for j,h in enumerate(saved['target_horizons']):
            e=P[:,j]-Y[:,j]
            for sp in ['train','validation','purged','test','new']:
                m=lab==sp; rows.append(dict(epoch=ep,h=h,split=sp,n200=int((np.abs(e[m])>200).sum()),mae=float(np.abs(e[m]).mean()),maxae=float(np.abs(e[m]).max())))
    name=f'{Path(EXT).name}__{run.name}'
    pd.DataFrame(rows).to_parquet(S/'scores'/f'{name}.parquet')
    np.savez_compressed(S/'scores'/f'{name}.npz',pred=np.stack(preds),y=Y,split=lab,anchor=A.astype('datetime64[ns]').astype(np.int64),epochs=np.array([int(p.parent.name.split('_')[1]) for p in eps]),horizons=np.array(saved['target_horizons']))
    print(name,len(eps),flush=True)
