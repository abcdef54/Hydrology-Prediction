"""Opt-in epoch diagnostics. No optimizer steps; never reads the test loader."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .change_objective import ChangeRiskLoss
from .losses import HorizonWeightedMSE, HorizonWeightedTailLoss
from .regional_training_data import regional_collate


@contextmanager
def isolated_evaluation(model):
    """Diagnostics must not change training randomness, modes or .grad buffers."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def objective_elements(trainer, predictions, targets, baseline, inputs, *, apply_rise_weight=True):
    """Weighted cells whose row-sum mean equals the configured data loss."""
    criterion = trainer.loss_func
    if isinstance(criterion, ChangeRiskLoss):
        return criterion.elements(predictions, targets, baseline, inputs['station_ids'],
                                  apply_rise_weight=apply_rise_weight)*criterion.weights.to(predictions)
    if isinstance(criterion, HorizonWeightedTailLoss):
        raise ValueError('Dynamics currently requires zero tail penalty')
    if isinstance(criterion, HorizonWeightedMSE):
        return (predictions-targets).square()*criterion.weights.to(predictions)
    if isinstance(criterion, nn.MSELoss) and criterion.reduction == 'mean':
        return (predictions-targets).square()/targets.shape[1]
    raise ValueError('Unsupported diagnostic objective')


def classification_elements(trainer, logits, targets, baseline, scales):
    labels = (((targets-baseline.detach())*scales) > trainer.loss_func.threshold_mm).to(logits.dtype)
    positive_weight = torch.as_tensor(trainer.rise_positive_weight, dtype=logits.dtype, device=logits.device)
    cells = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels,
                pos_weight=positive_weight, reduction='none')
    return cells*trainer.loss_func.weights.to(logits)*trainer.rise_classification_weight


class RegimeMeter:
    groups = ('all', 'extreme', 'rise', 'fall', 'non_extreme')
    fields = ('n', 'sse', 'sae', 'bias', 'actual_abs', 'pred_abs', 'persistence_sse',
              'large_errors', 'objective_sum', 'rise_n', 'rise_under_n', 'rise_shortfall',
              'alarms', 'hits', 'false_alarms', 'wrong_sign_n')

    def __init__(self, stations, horizons, device, threshold, error_threshold):
        self.values = torch.zeros(stations, horizons, len(self.groups), len(self.fields),
                                  dtype=torch.float64, device=device)
        self.threshold, self.error_threshold = threshold, error_threshold

    @torch.no_grad()
    def update(self, p, y, baseline, scale, ids, objective):
        error = (p.double()-y.double())*scale
        actual = (y.double()-baseline.double())*scale
        predicted = (p.double()-baseline.double())*scale
        extreme, rise = actual.abs() > self.threshold, actual > self.threshold
        fall, alarm = actual < -self.threshold, predicted > self.threshold
        masks = torch.stack((torch.ones_like(extreme), extreme, rise, fall, ~extreme), dim=-1)
        parts = torch.stack([value.double() for value in (
            torch.ones_like(error), error.square(), error.abs(), error, actual.abs(), predicted.abs(),
            actual.square(), error.abs() > self.error_threshold, objective,
            rise, rise & (error < 0), (-error).clamp_min(0)*rise,
            alarm, alarm & rise, alarm & ~rise, actual*predicted < 0)], dim=-1)
        self.values.index_add_(0, ids, parts.unsqueeze(2)*masks.unsqueeze(-1))

    def table(self, mapping, horizons, samples):
        values = self.values.cpu().numpy()
        rows = []
        def row(vector, level, sid, name, horizon, group):
            r = dict(zip(self.fields, vector))
            def ratio(a, b):
                return float(a/b) if b else float('nan')
            r.update(level=level, station_id=sid, station=name, horizon_hours=horizon, group=group,
                rmse_mm=ratio(r['sse'],r['n'])**.5, mae_mm=ratio(r['sae'],r['n']),
                bias_mm=ratio(r['bias'],r['n']), actual_abs_mm=ratio(r['actual_abs'],r['n']),
                predicted_abs_mm=ratio(r['pred_abs'],r['n']),
                persistence_skill=1-ratio(r['sse'],r['persistence_sse']),
                objective_contribution=r['objective_sum']/samples,
                rise_recall=ratio(r['hits'],r['rise_n']),
                rise_under_rate=ratio(r['rise_under_n'],r['rise_n']),
                rise_shortfall_mm=ratio(r['rise_shortfall'],r['rise_n']),
                rise_false_alarm_ratio=ratio(r['false_alarms'],r['alarms']),
                rise_false_positive_rate=ratio(r['false_alarms'],r['n']-r['rise_n']),
                wrong_sign_rate=ratio(r['wrong_sign_n'],r['n']))
            return r
        for sid, meta in mapping.items():
            for h, horizon in enumerate(horizons):
                for g, group in enumerate(self.groups):
                    rows.append(row(values[meta['station_index'],h,g], 'station_horizon', sid,
                                    meta.get('name',sid), horizon, group))
        pooled = values.sum(axis=0)
        for h, horizon in enumerate(horizons):
            for g, group in enumerate(self.groups):
                rows.append(row(pooled[h,g], 'pooled_horizon', 'pooled', 'All stations', horizon, group))
        for g, group in enumerate(self.groups):
            rows.append(row(pooled[:,g].sum(axis=0), 'pooled_all', 'pooled', 'All stations', 0, group))
        return pd.DataFrame(rows)


def fixed_probe_selection(dataset, threshold, maximum):
    """Train-only, deterministic strata; weight back to the true train mixture.

    Extreme anchor means at least one configured horizon exceeds the threshold.
    Each selected anchor gets population_stratum/(selected_stratum * N_train).
    Thus gradient norms describe estimated contributions to the original loss,
    not a deliberately event-balanced objective. Adjacent anchors are correlated.
    """
    if dataset.split != 'train':
        raise ValueError('Gradient probes must use training membership only')
    frame = dataset.view.samples
    columns = [f'target_h{h}_mm' for h in dataset.target_horizons]
    extreme = (frame[columns].to_numpy()-frame.water_now_mm.to_numpy()[:,None])
    extreme = (np.abs(extreme) > threshold).any(axis=1)
    rows = []
    for sid, group in frame.groupby('station_id',sort=False):
        positions = np.flatnonzero(frame.station_id.eq(sid).to_numpy())
        for flag in (False, True):
            candidates = positions[extreme[positions] == flag]
            budget = maximum//2 if flag else maximum-maximum//2
            selected = candidates[np.linspace(0,len(candidates)-1,min(budget,len(candidates)),dtype=int)]
            for index in selected:
                rows.append(dict(dataset_index=int(index), sample_id=int(frame.iloc[index].sample_id),
                    station_id=str(sid), station_index=int(frame.iloc[index].station_index),
                    anchor_utc=str(frame.iloc[index].anchor_utc), stratum='extreme' if flag else 'non_extreme',
                    population_n=len(candidates), objective_weight=len(candidates)/(len(selected)*len(frame))))
    return pd.DataFrame(rows)


class ChangeDynamics:
    def __init__(self, *, every=1, gradients=False, probe_samples=32):
        if every < 1 or probe_samples < 2:
            raise ValueError('Invalid diagnostic interval or probe budget')
        self.every, self.gradients, self.probe_samples = every, gradients, probe_samples
        self.probe_selection = None

    @staticmethod
    def loader(dataset, batch_size, device):
        # Never iterate the training loader or consume its sampler/worker RNG.
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                          collate_fn=regional_collate,
                          drop_last=False, pin_memory=device.type=='cuda',
                          generator=torch.Generator().manual_seed(7321))

    def evaluate_split(self, trainer, dataset, split, folder, epoch):
        mapping = dataset.regional_metadata['station_mapping']
        horizons = dataset.target_horizons
        threshold = getattr(trainer.loss_func, 'threshold_mm', trainer.validation_threshold_mm)
        meter = RegimeMeter(len(mapping),len(horizons),trainer.device,threshold,trainer.validation_threshold_mm)
        physical_batches, objective_batches, event_batches, offset = [], [], [], 0
        finite = torch.ones((), device=trainer.device, dtype=torch.bool)
        with torch.inference_mode():
            for batch in self.loader(dataset, trainer.train_dataloader.batch_size, trainer.device):
                inputs, targets, scales, means = trainer.prepare_batch(batch,dataset)
                if getattr(trainer.model, 'event_aware', False):
                    p, aux = trainer.model(**inputs, return_diagnostics=True)
                else:
                    p, aux = trainer.predict_batch(trainer.model,inputs), None
                base = trainer.predict_batch(trainer.model.persistence,inputs)
                cells = objective_elements(trainer,p,targets,base,inputs,apply_rise_weight=(split == 'train_eval'))
                if aux is not None and split == 'train_eval':
                    cells = cells + classification_elements(trainer,aux['rise_logits'],targets,base,scales)
                finite.logical_and_(torch.isfinite(p).all() & torch.isfinite(cells).all())
                meter.update(p,targets,base,scales,inputs['station_ids'],cells)
                if split == 'val_eval':
                    # Only compact outputs stay on the GPU, never history windows.
                    # Transfer once per split rather than synchronizing each batch.
                    physical_batches.append(torch.stack(
                        [(a.double()*scales+means) for a in (p,targets,base)], dim=-1))
                    objective_batches.append(cells)
                    if aux is not None:
                        event_batches.append(torch.stack([aux['rise_logits'], aux['rise_probability'],
                            aux['base_delta_normalized']*scales, aux['amplitude_normalized']*scales,
                            aux['correction_normalized']*scales], dim=-1))
                offset += len(targets)
        assert offset == len(dataset)
        if not finite.item():
            raise ValueError('Nonfinite diagnostic predictions or objective')
        table = meter.table(mapping,horizons,len(dataset))
        table.insert(0,'split',split)
        table.insert(0,'epoch',epoch)
        table.to_csv(folder/f'{split}_metrics.csv',index=False)
        if physical_batches:
            arrays = torch.cat(physical_batches).cpu().numpy()
            meta = dataset.view.samples
            # Repeated sample ID + horizon is the stable comparison key.
            frame = pd.DataFrame(dict(epoch=epoch, sample_id=np.repeat(meta.sample_id.to_numpy(),len(horizons)),
                station_id=np.repeat(meta.station_id.to_numpy(),len(horizons)),
                anchor_utc=np.repeat(meta.anchor_utc.to_numpy(),len(horizons)),
                horizon_hours=np.tile(horizons,len(meta)),
                predicted_mm=arrays[:,:,0].ravel(),actual_mm=arrays[:,:,1].ravel(),current_mm=arrays[:,:,2].ravel()))
            frame['target_utc'] = pd.to_datetime(frame.anchor_utc,utc=True)+pd.to_timedelta(frame.horizon_hours,unit='h')
            frame['error_mm'] = frame.predicted_mm-frame.actual_mm
            frame['actual_change_mm'] = frame.actual_mm-frame.current_mm
            frame['predicted_change_mm'] = frame.predicted_mm-frame.current_mm
            frame['objective_cell'] = torch.cat(objective_batches).cpu().numpy().ravel()
            if event_batches:
                event = torch.cat(event_batches).cpu().numpy().reshape(-1, 5)
                for index, name in enumerate(('rise_logit', 'rise_probability', 'base_delta_mm',
                                               'amplitude_mm', 'correction_mm')):
                    frame[name] = event[:, index]
            frame.to_parquet(folder/'validation_predictions.parquet',index=False)
        return table

    def gradient_probe(self, trainer, folder):
        data = trainer.train_dataloader.dataset
        if self.probe_selection is None:
            self.probe_selection = fixed_probe_selection(data,
                getattr(trainer.loss_func,'threshold_mm',trainer.validation_threshold_mm),self.probe_samples)
        self.probe_selection.to_csv(folder.parent/'gradient_samples.csv',index=False)
        parameters = [p for p in trainer.model.encoder.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError('Gradient probes require a trainable shared encoder')
        vectors, rows = {}, []
        # cuDNN RNNs cannot backpropagate an eval-mode forward. Use the native
        # RNN path for this small probe; keep dropout disabled and restore flags.
        with torch.backends.cudnn.flags(enabled=False), torch.inference_mode(False), torch.enable_grad():
            for sid, group in self.probe_selection.groupby('station_id',sort=False):
                loader = self.loader(Subset(data,group.dataset_index.tolist()),len(group),trainer.device)
                batch = next(iter(loader))
                inputs,y,scales,_ = trainer.prepare_batch(batch,data,need_mean=False)
                if getattr(trainer.model, 'event_aware', False):
                    p, aux = trainer.model(**inputs, return_diagnostics=True)
                else:
                    p, aux = trainer.predict_batch(trainer.model,inputs), None
                base = trainer.predict_batch(trainer.model.persistence,inputs)
                cells = objective_elements(trainer,p,y,base,inputs)
                if aux is not None:
                    cells = cells + classification_elements(trainer,aux['rise_logits'],y,base,scales)
                weights = torch.as_tensor(group.objective_weight.to_numpy(copy=True),device=p.device,dtype=p.dtype)
                loss = (cells.sum(dim=1)*weights).sum()
                grads = torch.autograd.grad(loss,parameters,allow_unused=True)
                vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                    for p,g in zip(parameters,grads)]).double().cpu()
                if not torch.isfinite(vector).all():
                    raise ValueError('Nonfinite diagnostic gradient')
                vectors[sid] = vector
                rows.append(dict(station_id=sid, sample_count=len(group), encoder_gradient_norm=vector.norm().item(),
                                 estimated_objective_contribution=loss.detach().item()))
        pairs = []
        for i, (sid,a) in enumerate(vectors.items()):
            for other,b in list(vectors.items())[i+1:]:
                denominator = (a.norm()*b.norm()).item()
                cosine = float(np.clip((a@b).item()/denominator,-1.,1.)) if denominator else float('nan')
                pairs.append(dict(station_a=sid,station_b=other,cosine=cosine,
                                  valid=bool(denominator),parameter_scope='encoder',
                                  status='measured' if denominator else 'zero_gradient'))
        pd.DataFrame(rows).to_csv(folder/'gradient_norms.csv',index=False)
        pd.DataFrame(pairs,columns=['station_a','station_b','cosine','valid','parameter_scope','status']).to_csv(
            folder/'gradient_cosines.csv',index=False)

    @staticmethod
    def fingerprint(dataset):
        columns = ['sample_id','station_id','anchor_utc','water_now_mm']+[
            f'target_h{h}_mm' for h in dataset.target_horizons]
        content = dataset.view.samples[columns].to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()

    def run(self, trainer, epoch):
        # Includes MLflow calls: even logging retries must not consume RNG used
        # by training. The nested split/probe work never touches .grad buffers.
        with isolated_evaluation(trainer.model):
            return self._run(trainer,epoch)

    def _run(self, trainer, epoch):
        data = trainer.train_dataloader.dataset
        validation = trainer.eval_dataloader.dataset
        if not getattr(trainer.model,'num_stations',0) or not hasattr(data,'view'):
            raise ValueError('Dynamics requires regional station conditioning')
        if data.split != 'train' or validation.split != 'validation':
            raise ValueError('Diagnostics only accept train and validation datasets')
        if getattr(trainer,'correction_penalty',0) or isinstance(trainer.loss_func,HorizonWeightedTailLoss):
            raise ValueError('Dynamics requires zero correction/tail penalties')
        root = trainer.checkpoint_path.with_suffix('.dynamics')
        folder = root/f'epoch_{epoch:03d}'
        folder.mkdir(parents=True,exist_ok=False)
        with isolated_evaluation(trainer.model):
            snapshot = folder/'model.pt'
            trainer.save_checkpoint(snapshot)
            trainer.save_preprocessing(snapshot,announce=False)
            tables = {split:self.evaluate_split(trainer,dataset,split,folder,epoch)
                      for split,dataset in [('train_eval',data),('val_eval',validation)]}
            if self.gradients:
                self.gradient_probe(trainer,folder)
        metadata = dict(format_version=1,epoch=epoch,status='complete',checkpoint='model.pt',
            selected_best_epoch=trainer.best_checkpoint_epoch,best_val_loss=trainer.best_val_loss,
            configured_epochs=trainer.num_epochs,
            training_finished=bool(epoch>=trainer.num_epochs or (trainer.early_stopping and
                trainer.epochs_without_improvement>=trainer.early_stopping)),
            evaluation_mode='eval, dropout disabled; weights fixed',
            validation_fingerprint=self.fingerprint(validation),train_fingerprint=self.fingerprint(data),
            dataset_manifest_sha256=data.regional_metadata.get('dataset_manifest_sha256'),
            target_horizons=data.target_horizons,time_features=data.regional_metadata.get('time_features'),
            head_mode=trainer.model.head_mode,
            horizon_decoder=getattr(trainer.model,'horizon_decoder_mode','none'),
            loss=trainer.loss_func.specification() if isinstance(trainer.loss_func,ChangeRiskLoss) else type(trainer.loss_func).__name__,
            gradient_probe=dict(enabled=self.gradients,maximum_samples_per_station=self.probe_samples,
                scope='encoder',sampling='Fixed train-only extreme/non-extreme anchors, population reweighted',
                caveat='Correlated, finite probes; a negative cosine is not proof of harmful transfer'))
        (folder/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        # A small separate namespace; never overwrite checkpoint-selection metrics.
        import mlflow
        metrics = {}
        for split, table in tables.items():
            for group in RegimeMeter.groups:
                row = table.loc[table.level.eq('pooled_all')&table.group.eq(group)].iloc[0]
                for field in ('rmse_mm','mae_mm','bias_mm','rise_recall','rise_false_alarm_ratio','objective_contribution'):
                    if np.isfinite(row[field]):
                        metrics[f'dynamics/{split}/{group}/{field}'] = float(row[field])
        if mlflow.active_run():
            mlflow.log_metrics(metrics,step=epoch)
        print(f'Dynamics epoch {epoch}: {folder} (train/validation eval only)')
        return folder
