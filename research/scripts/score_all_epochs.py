"""Score every epoch of one run on validation, test and the unseen part of an extended dataset (> 2026-10-14 19:00 UTC).

Usage: PYTHONPATH=. .venv/bin/python research/scripts/score_all_epochs.py OUT.parquet TRAIN_DATASET_DIR EXT_DATASET_DIR RUN_DIR
Output: one row per (epoch, split, horizon): n200, under, over, maxae, mae. Uses the GPU when available.
"""
import sys, joblib, numpy as np, pandas as pd, torch
from pathlib import Path
from src.ai.regional_training_data import regional_loaders
from src.ai.sequence_checkpoint import SequenceRuntime
torch.set_num_threads(4); DEV='cuda' if torch.cuda.is_available() else 'cpu'
OUT, DS, EXT, RUN = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
eps=sorted(p for p in RUN.rglob('epoch_*/model.pt') if p.parent.name!='epoch_000')
saved=joblib.load(eps[0].with_suffix('.preprocessing.joblib'))['regional']
kw=dict(batch_size=512,num_workers=0,seed=0,rain_stations='+'.join(saved['rain_stations']) or 'none',use_weather_forecast=False,
        time_features=saved['time_features'],target_horizons=saved['target_horizons'],sequence_length=saved['training_selection']['sequence_length'])
ld=regional_loaders(DS,**kw); le=regional_loaders(EXT,**kw)
def prep(loader,mask=None):
    fr=loader.dataset.view.samples; b=list(loader); return b,fr
data={'validation':prep(ld[1]),'test':prep(ld[2]),'new':prep(le[2])}
start=pd.Timestamp('2026-10-14 19:00',tz='UTC')
rows=[]
for p in eps:
    rt=SequenceRuntime(p, device=DEV); ep=int(p.parent.name.split('_')[1])
    for sp,(batches,fr) in data.items():
        pred=np.concatenate([rt.predict_mm(b['x'],b['station_ids']) for b in batches])
        a=pd.to_datetime(fr.anchor_utc,utc=True).to_numpy()
        keep=np.ones(len(fr),bool) if sp!='new' else (pd.DatetimeIndex(a)>start)
        for j,h in enumerate(saved['target_horizons']):
            e=pred[keep,j]-fr[f'target_h{h}_mm'].to_numpy()[keep]
            rows.append(dict(epoch=ep,split=sp,h=h,n200=int((np.abs(e)>200).sum()),under=int((e<-200).sum()),over=int((e>200).sum()),maxae=float(np.abs(e).max()),mae=float(np.abs(e).mean())))
pd.DataFrame(rows).to_parquet(OUT); print(RUN.name, len(eps))
