"""Tree forecasts over explicit past-hour lags, with physical-mm validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb


def sequence_table(frame,features,targets,sequence_length):
    """Same warmup and complete-target population as LSTM; predictors retain NaN."""
    if sequence_length<1:raise ValueError('Sequence length must be positive')
    times=pd.to_datetime(frame.timestamp_utc,utc=True)
    if not times.is_unique or not times.is_monotonic_increasing or not times.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Ordered complete hourly grid required')
    if any(c.startswith('target_') or c.startswith('timestamp') for c in features):
        raise ValueError('Features cannot contain targets or timestamps')
    raw=frame[features].to_numpy(dtype=np.float32)
    labels=frame[targets].to_numpy(dtype=np.float64)
    if np.isinf(raw).any():raise ValueError('Infinite feature values')
    if len(frame)<sequence_length:raise ValueError('Insufficient history')
    level=frame.water_level_depth.to_numpy(dtype=np.float64)
    ends=np.arange(sequence_length-1,len(frame))
    ends=ends[np.isfinite(labels[ends]).all(axis=1)&np.isfinite(level[ends])]
    # Newest timestep first; every raw column, including forecast/masks, is lagged.
    matrix=np.concatenate([raw[ends-lag] for lag in range(sequence_length)],axis=1)
    names=[f'lag{lag:03d}__{name}' for lag in range(sequence_length) for name in features]
    return dict(X=matrix,y=labels[ends],level=level[ends],indices=ends,
                times=times.iloc[ends].reset_index(drop=True),names=names)


class ValidationSelector:
    def __init__(self,metric,patience,baseline_errors):
        if metric not in ['large_error_rate','rmse'] or patience<1:raise ValueError('Invalid checkpoint rule')
        self.metric=metric;self.patience=patience;self.history=[]
        self.best=self.score(int((abs(baseline_errors)>200).sum()),float(np.sqrt(np.mean(baseline_errors**2))))
        self.best_metrics=dict(above_200=int((abs(baseline_errors)>200).sum()),rmse_mm=float(np.sqrt(np.mean(baseline_errors**2))))
        self.best_round=0

    def score(self,n200,rmse):
        if not np.isfinite(rmse):raise ValueError('Non-finite validation metric')
        return (int(n200),float(rmse)) if self.metric=='large_error_rate' else (float(rmse),)

    def update(self,round_number,n200,rmse):
        score=self.score(n200,rmse)
        self.history.append(dict(round=round_number,above_200=int(n200),rmse_mm=float(rmse)))
        if score<self.best:
            self.best=score;self.best_round=round_number
            self.best_metrics=dict(above_200=int(n200),rmse_mm=float(rmse))
        return round_number-self.best_round>=self.patience


def train_horizon(method,train,val,j,options,output,on_iteration=None):
    """One independent booster per horizon; restore validation-selected round."""
    residual=options['target_mode']=='residual'
    y_train=train['y'][:,j]-(train['level'] if residual else 0)
    y_val=val['y'][:,j]-(val['level'] if residual else 0)
    selector=ValidationSelector(options['checkpoint_metric'],options['early_stopping'],val['level']-val['y'][:,j])
    common=dict(learning_rate=options['learning_rate'],max_depth=options['max_depth'],
        seed=options['seed'])
    if method=='xgb':
        physical={}
        def metric(pred,dataset):
            error=pred.astype(np.float64)-y_val
            physical['count']=int((abs(error)>200).sum())
            physical['rmse']=float(np.sqrt(np.mean(error**2)))
            return 'large_error_count',float(physical['count'])
        class Stop(xgb.callback.TrainingCallback):
            def after_iteration(self,model,epoch,evals_log):
                stop=selector.update(epoch+1,physical['count'],physical['rmse'])
                if on_iteration is not None:on_iteration(epoch+1,selector)
                return stop
        params=dict(**common,objective='reg:squarederror',eval_metric='rmse',tree_method='hist',
            device=options['device'],nthread=options['n_jobs'],max_bin=128,
            subsample=options.get('subsample',.8),colsample_bytree=options.get('colsample_bytree',.8),
            min_child_weight=options.get('min_child_weight',5.),reg_lambda=options.get('reg_lambda',1.),reg_alpha=options.get('reg_alpha',0.))
        dtrain=xgb.DMatrix(train['X'],label=y_train);dval=xgb.DMatrix(val['X'],label=y_val)
        booster=xgb.train(params,dtrain,num_boost_round=options['n_estimators'],evals=[(dval,'val')],
            custom_metric=metric,
            callbacks=[Stop()],verbose_eval=False)
        if selector.best_round:
            booster=booster[:selector.best_round];booster.save_model(output)
    elif method=='lgbm':
        if options['device']!='cpu':raise ValueError('LightGBM runner uses CPU; GPU build support is not assumed')
        def stop(env):
            values={r[1]:r[2] for r in env.evaluation_result_list if r[0]=='val'}
            done=selector.update(env.iteration+1,values['large_error_count'],values['rmse_mm'])
            if on_iteration is not None:on_iteration(env.iteration+1,selector)
            if done:
                raise lgb.callback.EarlyStopException(max(selector.best_round-1,0),env.evaluation_result_list)
        stop.order=30
        params=dict(**common,objective='regression',metric='rmse',verbosity=-1,device_type='cpu',
            num_threads=options['n_jobs'],num_leaves=options['num_leaves'],min_data_in_leaf=options.get('min_data_in_leaf',20),
            bagging_fraction=options.get('subsample',.8),bagging_freq=1,feature_fraction=options.get('colsample_bytree',.8),
            lambda_l2=options.get('reg_lambda',1.),lambda_l1=options.get('reg_alpha',0.),max_bin=127,
            deterministic=True,force_col_wise=True)
        dtrain=lgb.Dataset(train['X'],label=y_train)
        dval=lgb.Dataset(val['X'],label=y_val,reference=dtrain)
        booster=lgb.train(params,dtrain,num_boost_round=options['n_estimators'],valid_sets=[dval],valid_names=['val'],
            feval=lambda p,d:[('large_error_count',float((abs(p-y_val)>200).sum()),False),
                ('rmse_mm',float(np.sqrt(np.mean((p-y_val)**2))),False)],callbacks=[stop])
        if selector.best_round:booster.save_model(str(output),num_iteration=selector.best_round)
    else:raise ValueError('Unknown tree method')
    return dict(kind='tree' if selector.best_round else 'persistence',best_round=selector.best_round,
        validation_score=selector.best,validation_metrics=selector.best_metrics,
        history=selector.history,path=Path(output).name if selector.best_round else None)


class TreeSequenceForecast:
    """Native JSON/text artifacts; no tree model pickle or MLflow model upload."""
    def __init__(self,directory):
        self.directory=Path(directory)
        self.spec=json.loads((self.directory/'tree_model.json').read_text())
        if (self.spec.get('status')!='complete'
                or len(self.spec['horizons'])!=len(self.spec['target_columns'])
                or [h['target'] for h in self.spec['horizons']]!=self.spec['target_columns']):
            raise ValueError('Incomplete or inconsistent tree artifact')
        self.models=[]
        for item in self.spec['horizons']:
            if item['kind']=='persistence':self.models.append(None)
            elif self.spec['train_method']=='xgb':
                model=xgb.Booster();model.load_model(self.directory/item['path']);model.set_param({'device':'cpu'});self.models.append(model)
            else:self.models.append(lgb.Booster(model_file=str(self.directory/item['path'])))

    def predict_table(self,data):
        result=[]
        for model in self.models:
            if model is None:prediction=data['level'].copy()
            else:
                p=(model.predict(xgb.DMatrix(data['X'])) if self.spec['train_method']=='xgb' else model.predict(data['X']))
                prediction=p+(data['level'] if self.spec['target_mode']=='residual' else 0)
            result.append(prediction)
        values=np.column_stack(result)
        if not np.isfinite(values).all():raise ValueError('Non-finite tree predictions')
        return values

    def predict_frame(self,frame):
        data=sequence_table(frame,self.spec['feature_columns'],self.spec['target_columns'],self.spec['sequence_length'])
        return data,self.predict_table(data)
