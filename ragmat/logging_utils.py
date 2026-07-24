from __future__ import annotations
import json, logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).parent.parent
_LOGS_DIR = _PROJECT_ROOT / 'logs'

def _ensure_logs_dir():
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _write_entry(log_file, entry):
    _ensure_logs_dir()
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=str) + chr(10))

class AnomalyLogger:
    @staticmethod
    def log_anomaly(check_id, msg, data=None):
        entry = {'timestamp': datetime.utcnow().isoformat()+'Z','type':'ANOMALY','check_id':check_id,'msg':msg,'data':data or {}}
        _write_entry(_LOGS_DIR / 'anomalies.log', entry)
        logger.warning('ANOMALY [%s]: %s', check_id, msg)
    @staticmethod
    def log_h1_status(status, run_id, data=None):
        entry = {'timestamp': datetime.utcnow().isoformat()+'Z','type':'H1_STATUS','status':status,'run_id':run_id,'data':data or {}}
        _write_entry(_LOGS_DIR / 'h1_status.log', entry)
        logger.info('H1_STATUS [%s] run=%s', status, run_id)
    @staticmethod
    def log_critical_failure(msg, data=None):
        entry = {'timestamp': datetime.utcnow().isoformat()+'Z','type':'CRITICAL_FAILURE','msg':msg,'data':data or {}}
        _write_entry(_LOGS_DIR / 'critical_failures.log', entry)
        logger.critical('CRITICAL FAILURE: %s', msg)
        raise RuntimeError('CRITICAL FAILURE -- see logs/critical_failures.log: ' + msg)

def check_ood_scores_valid(ood_scores, experiment_name):
    import numpy as np
    if float(np.std(ood_scores)) == 0.0:
        AnomalyLogger.log_critical_failure('OOD scores std=0: detector stubbed or not fitted.', {'experiment_name': experiment_name})

def check_low_ood_not_all(ood_scores, split_type, experiment_name, threshold=1.0):
    import numpy as np
    if split_type not in ('family_out', 'element_out'):
        return
    frac = float((ood_scores < threshold).mean())
    if frac >= 0.99:
        AnomalyLogger.log_anomaly('LOW_OOD_ALL', 'low_ood==all on '+split_type+': OOD detector may not fire.', {'experiment_name':experiment_name,'split_type':split_type,'frac':frac,'threshold':threshold})

def check_mae_range(mae_val, split_type, property_name, experiment_name):
    BOUNDS = {('formation_energy','iid'):(0.08,0.15),('formation_energy','family_out'):(0.12,0.45),('formation_energy','element_out'):(0.15,0.55),('band_gap','iid'):(0.15,0.55),('band_gap','family_out'):(0.12,0.65),('band_gap','element_out'):(0.15,0.75)}
    key = (property_name, split_type)
    if key not in BOUNDS: return
    lo, hi = BOUNDS[key]
    if not (lo <= mae_val <= hi):
        AnomalyLogger.log_anomaly('MAE_OUT_OF_RANGE', 'MAE='+str(round(mae_val,5))+' outside ['+str(lo)+','+str(hi)+'] for '+property_name+'/'+split_type, {'experiment_name':experiment_name,'mae':mae_val,'lo':lo,'hi':hi})

def check_retrieval_differs_from_baseline(retrieval_mae, baseline_mae, experiment_name, tol=1e-3):
    if abs(retrieval_mae - baseline_mae) < tol:
        AnomalyLogger.log_anomaly('M6_MAE_IDENTICAL','Retrieval MAE identical to baseline -- possible concat passthrough bug.',{'experiment_name':experiment_name,'retrieval_mae':retrieval_mae,'baseline_mae':baseline_mae,'diff':abs(retrieval_mae-baseline_mae)})

def check_n_samples(n_actual, n_expected, experiment_name, tolerance=100):
    if n_actual == 0:
        AnomalyLogger.log_critical_failure('n_samples=0 in results for '+experiment_name, {'n_actual':n_actual,'n_expected':n_expected})
    if abs(n_actual - n_expected) > tolerance:
        AnomalyLogger.log_anomaly('N_SAMPLES_MISMATCH','n_samples='+str(n_actual)+' differs from expected '+str(n_expected),{'experiment_name':experiment_name,'n_actual':n_actual,'n_expected':n_expected})

def check_result_file_collision(result_path, experiment_name):
    if Path(result_path).exists():
        AnomalyLogger.log_anomaly(
            'RESULT_FILE_COLLISION',
            'Result file already exists: ' + str(result_path) + '. Will overwrite.',
            {'experiment_name': experiment_name, 'path': str(result_path)}
        )
