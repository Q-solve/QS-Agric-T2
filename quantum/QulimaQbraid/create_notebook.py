import json
from textwrap import dedent

cells = []


def md(s):
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": dedent(s).splitlines(True)}
    )


def code(s):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": dedent(s).splitlines(True),
        }
    )


md(
    """# Hybrid quantum-classical maize yield forecasting\n\nThis notebook builds a county-year modeling table for Kenya, keeps rainfall/weather features separate from contextual features, uses a leakage-safe chronological split, and fits a quantum-kernel residual model. The quantum model is intentionally small (4 contextual features) so it is practical on a local simulator."""
)
code(
    """from pathlib import Path\nimport numpy as np, pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler\nfrom sklearn.linear_model import Ridge\nfrom sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nfrom sklearn.model_selection import TimeSeriesSplit\n\nDATA = Path('/home/jovyan/qulima_maize_yield.csv')\nOUT = DATA.parent / 'maize_quantum_outputs'; OUT.mkdir(exist_ok=True)\ndf = pd.read_csv(DATA)\nprint('shape:', df.shape)\ndf.head()"""
)
code("""print('Columns and dtypes:')\nprint(df.dtypes.to_string())\nprint('\\nCounty count:', df.county.nunique(), sorted(df.county.unique()))\nprint('Years:', df.year.min(), 'to', df.year.max(), 'unique:', df.year.nunique())\nprint('Season basis:', df.season_basis.value_counts(dropna=False).to_dict())\nmissing = df.isna().sum()
print('Missing values:', missing[missing.gt(0)].to_dict())\nprint(df.describe(include='all').T[['count','mean','min','max']].to_string())""")
md(
    """## Modeling table\n\nThe source contains annual observations (rather than separate seasonal rows). The `season_basis` field is retained; when March–June observations are supplied, the same code will filter to them. Rainfall/weather variables are explicitly separated from non-rainfall contextual variables. Source URL columns and post-harvest production are excluded to avoid target leakage."""
)
code(
    """# Prefer March-June seasonal rows if present; otherwise use the annual county-year observations supplied.\nseason_mask = df['season_basis'].astype(str).str.lower().str.contains('mar|march|m-j|mj|season')\nif season_mask.any():\n    model = df.loc[season_mask].copy()\n    season_note = 'Filtered to March-June/season rows'\nelse:\n    model = df.copy()\n    season_note = 'No March-June rows found; using annual county-year rows'\nmodel = model.sort_values(['year','county']).reset_index(drop=True)\nRAIN = ['rain_dekads_available','rain_total_jan_jun_mm','rain_lta_jan_jun_mm','rain_anomaly_pct','rain_mean_dekad_mm','rain_std_dekad_mm','rain_cv','wet_dekads_ge20mm','heavy_dekads_ge50mm','dry_dekads_lt10mm','longest_dry_spell_dekads','rain_onset_doy']\nCONTEXT = ['county_lat','county_lon','maize_area_ha']\nTARGET='yield_t_ha'\nkeep=['county','county_pcode','year','season_basis',TARGET]+RAIN+CONTEXT\nmodel_table=model[keep].copy()\nprint(season_note); print(model_table.shape); display(model_table.head())\nmodel_table.to_csv(OUT/'county_year_modeling_table.csv',index=False)"""
)
md(
    """## Leakage-safe temporal validation\n\nThe final test is the latest 20% of years. Training-side medians and all fitted models are learned only from the earlier period. A rainfall-only Ridge is the operational baseline; contextual classical and quantum models predict residuals relative to it."""
)
code(
    """years=np.sort(model_table.year.unique()); cutoff=years[max(1,int(np.ceil(len(years)*.8))-1)]\ntrain=model_table[model_table.year<=cutoff].copy(); test=model_table[model_table.year>cutoff].copy()\nif test.empty: # robust fallback for very short data\n    cutoff=years[-2]; train=model_table[model_table.year<=cutoff].copy(); test=model_table[model_table.year>cutoff].copy()\nprint('train years',train.year.min(),train.year.max(),'test years',test.year.min(),test.year.max(),'rows',len(train),len(test))\n\ndef fit_predict(cols, estimator=Ridge(alpha=1.0)):\n    pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',estimator)])\n    pipe.fit(train[cols],train[TARGET]); return pipe, pipe.predict(test[cols])\nrain_pipe,rain_pred=fit_predict(RAIN)\ntrain_rain_pred=rain_pipe.predict(train[RAIN]); train_resid=train[TARGET].to_numpy()-train_rain_pred\nprint('Rainfall baseline RMSE/MAE/R2:', mean_squared_error(test[TARGET],rain_pred)**.5, mean_absolute_error(test[TARGET],rain_pred), r2_score(test[TARGET],rain_pred))"""
)
code(
    """# Classical comparators and residual contextual model\nctx=['county_lat','county_lon','maize_area_ha']\nctx_pipe,ctx_pred=fit_predict(ctx,RandomForestRegressor(n_estimators=250,min_samples_leaf=4,random_state=7,n_jobs=-1))\nall_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',Ridge(alpha=10))])\nall_pipe.fit(train[RAIN+ctx],train[TARGET]); all_pred=all_pipe.predict(test[RAIN+ctx])\nmodels={'rainfall_ridge':rain_pred,'context_rf':ctx_pred,'all_features_ridge':all_pred}\nfor n,p in models.items(): print(n, 'RMSE %.4f MAE %.4f R2 %.4f'%(mean_squared_error(test[TARGET],p)**.5,mean_absolute_error(test[TARGET],p),r2_score(test[TARGET],p)))"""
)
md(
    """## Quantum residual model\n\nWe use four contextual features (`county_lat`, `county_lon`, `maize_area_ha`, and a deterministic county code) and map each to `[0, π]`. The `FidelityQuantumKernel` computes a train/test Gram matrix. Kernel Ridge regression is then fit to the rainfall-baseline residuals. This is a quantum-kernel residual learner, not a claim that the quantum model is universally superior."""
)
code(
    """# Small contextual subset (4 features), with county identity represented numerically and fitted only from train.\nall_counties=sorted(model_table.county.astype(str).unique()); county_code={c:i/(max(1,len(all_counties)-1)) for i,c in enumerate(all_counties)}\nfor x in (train,test): x['county_code']=x.county.astype(str).map(county_code)\nQ=['county_lat','county_lon','maize_area_ha','county_code']\nqscale=Pipeline([('impute',SimpleImputer(strategy='median')),('angle',MinMaxScaler(feature_range=(0,np.pi)))])\nXtr=qscale.fit_transform(train[Q]); Xte=qscale.transform(test[Q])\nfrom qiskit.circuit.library import ZZFeatureMap\nfrom qiskit_machine_learning.kernels import FidelityQuantumKernel\nfeature_map=ZZFeatureMap(feature_dimension=len(Q),reps=2,entanglement='linear')\nqkernel=FidelityQuantumKernel(feature_map=feature_map)\nKtr=qkernel.evaluate(Xtr); Kte=qkernel.evaluate(Xte,Xtr)\nfrom sklearn.kernel_ridge import KernelRidge\nqmodel=KernelRidge(alpha=1.0,kernel='precomputed').fit(Ktr,train_resid)\nq_resid=qmodel.predict(Kte); quantum_pred=rain_pred+q_resid\nmodels['quantum_residual']=quantum_pred\nprint('quantum kernel shape:',Ktr.shape)\nfor n,p in models.items(): print(n, 'RMSE %.4f MAE %.4f R2 %.4f'%(mean_squared_error(test[TARGET],p)**.5,mean_absolute_error(test[TARGET],p),r2_score(test[TARGET],p)))"""
)
code(
    """metrics=[]\nfor n,p in models.items(): metrics.append({'model':n,'RMSE':mean_squared_error(test[TARGET],p)**.5,'MAE':mean_absolute_error(test[TARGET],p),'R2':r2_score(test[TARGET],p)})\nmetrics=pd.DataFrame(metrics).sort_values('RMSE'); display(metrics); metrics.to_csv(OUT/'model_metrics.csv',index=False)\npred_out=test[['county','year',TARGET]].copy()\nfor n,p in models.items(): pred_out[n+'_prediction']=p\npred_out.to_csv(OUT/'test_predictions.csv',index=False)\nfig,ax=plt.subplots(1,2,figsize=(14,5))\nmetrics.set_index('model')['RMSE'].plot.bar(ax=ax[0],title='Test RMSE (lower is better)',color='steelblue'); ax[0].tick_params(axis='x',rotation=30)\nax[1].scatter(test[TARGET],quantum_pred,alpha=.65,label='quantum residual'); lo,hi=test[TARGET].min(),test[TARGET].max(); ax[1].plot([lo,hi],[lo,hi],'k--'); ax[1].set(xlabel='Actual yield (t/ha)',ylabel='Predicted yield (t/ha)',title='Quantum residual predictions'); ax[1].legend()\nplt.tight_layout(); fig.savefig(OUT/'model_comparison.png',dpi=160); display(fig)\nprint('Saved outputs to',OUT)"""
)
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "maize-quantum",
            "language": "python",
            "name": "maize-quantum",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
try:
    with open("/home/jovyan/maize-quantum-forecasting.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
except OSError as exc:
    raise RuntimeError(f"Unable to write notebook: {exc}") from exc
print("wrote notebook")
