from core.utils_analysis import getPredAggSurv_GBMLGG, CI_pm
from core.utils_analysis import getPredAggGrad_GBMLGG, calcGradMetrics, calcAggGradMetrics
from tqdm import tqdm

import numpy as np
import pandas as pd


def surv_evaluation():
    """
    Evaluate the c-index for the survival prediction task.
    """
    models = ['pathomic_late_0.01CRD']
    model_names = ['Pathomic F. (CNN+SNN)', 'Histology CNN', 'Genomic SNN']
    # models = ['omic', 'path', 'pathomic_fusion']
    # model_names = ['Genomic SNN', 'Histology CNN', 'Pathomic F. (CNN+SNN)', 'Pathomic F. CNN', 'Pathomic F. SNN']

    ckpt_name = './checkpoints/TCGA_GBMLGG/surv_15_rnaseq/'
    # cv_surv = [np.array(getPredAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model)) for model in tqdm(models)]
    # cv_surv = pd.DataFrame(np.array(cv_surv))
    # cv_surv = [np.array(getPredAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model)) for model in tqdm(models)]
    cv_surv = None
    for model in tqdm(models):
        results = np.array(getPredAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model))
        cv_surv = results if cv_surv is None else np.concatenate((cv_surv, results), axis=0)
        # print(results)
        print("cv_surv:", cv_surv)
    cv_surv = pd.DataFrame(cv_surv)
    print(cv_surv)
    cv_surv.columns = ['Split %s' % str(k) for k in range(1,16)]
    cv_surv.index = model_names
    cv_surv['C-Index'] = [CI_pm(cv_surv.loc[model]) for model in model_names]

def grad_evaluation():
    """
    Compute the matrics for the grading task.
    """
    # models = ['pathomic_self_MT_5cv_0118']
    # models = ['0322_pofusion_path_omic_4views_tsvd_lam0.1']
    # models = ['0330_pofusion_MT']
    # models = ['0404_concat_fusion']
    # models = ['pathomic_LMF_fusion_MT_0315']
    # models = ["0407_pofusion_seed2021"]
    models = ["0407_tsvd_4views_lam0.1_seed2021"]
    model_names = ['Pathomic F. (CNN+SNN)']
    # model_names = ['Histology CNN']
    # model_names = ['Genomics SNN']
    ckpt_name = './checkpoints/TCGA_GBMLGG/grad_15/'

    grad_all_splits, cv_grad = None, None
    
    for model in tqdm(models):
        print("evaluating")
        ## "which_net: [fuse, path, omic]"
        y_label, y_pred = getPredAggGrad_GBMLGG(ckpt_name = ckpt_name, model=model, agg_type='max', which_net = "fuse")
        print("number of patients:", [y_label[i].shape for i in range(len(y_label))], [y_pred[i].shape for i in range(len(y_pred))])
        cv_grad_split = np.array([calcGradMetrics(y_label, y_pred)])
        grad_all_splits = cv_grad_split if grad_all_splits is None \
            else np.concatenate((grad_all_splits, cv_grad_split), axis=0)

        grad_results = np.array([calcAggGradMetrics(y_label, y_pred)])
        cv_grad = grad_results if cv_grad is None else np.concatenate((cv_grad, grad_results), axis=0)               
    
    grad_all_splits = pd.DataFrame(grad_all_splits)
    grad_all_splits.index = model_names
    grad_all_splits.columns = ['Split %s' % str(k) for k in range(1,6)]
    print(grad_all_splits)

    cv_grad = pd.DataFrame(cv_grad)
    cv_grad.index = model_names
    cv_grad.columns = ['AUC', 'AP', 'F1', 'F1 Grade IV']
    print(cv_grad)


if __name__ == "__main__":
    # surv_evaluation()
    grad_evaluation()