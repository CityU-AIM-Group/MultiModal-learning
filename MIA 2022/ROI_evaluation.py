import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelBinarizer
from core.utils_analysis import calcGradMetrics, calcAggGradMetrics

def grad_evaluation(which_net="fuse"):
    """
    Compute the matrics for the grading task.
    """
    # models = ['pathomic_self_MT_5cv_0118']
    # models = ['0322_pofusion_path_omic_4views_tsvd_lam0.1']
    models = ['0330_pofusion_MT']
    # models = ['0404_concat_fusion']
    # models = ['pathomic_LMF_fusion_MT_0315']
    # models = ["0407_pofusion_seed2021"]
    # models = ["0407_tsvd_4views_lam0.1_seed2021"]
    model_names = ['Pathomic F. (CNN+SNN)']
    # model_names = ['Histology CNN']
    # model_names = ['Genomics SNN']
    ckpt_name = './checkpoints/TCGA_GBMLGG/grad_15/'

    grad_all_splits, cv_grad = None, None
    y_label, y_pred = [], []
    
    print("evaluating")
    for model in models:
        for k in range(1,6):
            ### Loads Prediction Pickle File. Registers predictions with TCGA IDs for the test split.
            pred = pickle.load(open(ckpt_name+'/%s/%s_%d_patch_pred_test.pkl' % (model, model, k), 'rb'))  
            if which_net == "fuse":
                pred_idx = 5
            elif which_net == "path":
                pred_idx = 6
            elif which_net == "omic":
                pred_idx = 7
            grad_pred = pred[pred_idx].T
            grad_pred = pd.DataFrame(np.stack(grad_pred)).T
            grad_pred.columns = ['score_0', 'score_1', 'score_2']
            grad_gt = pred[8]

            grad_pred = np.array(grad_pred)
            enc = LabelBinarizer()
            enc.fit(grad_gt)
            grad_gt = enc.transform(grad_gt)

            ### 合并成ROI级别的prediction和ground truth
            grad_pred = np.reshape(np.expand_dims(grad_pred, 1), (-1, 9, 3))
            grad_pred = np.squeeze(np.mean(grad_pred, 1))

            grad_gt = np.reshape(np.expand_dims(grad_gt, 1), (-1, 9, 3))
            grad_gt = np.squeeze(np.mean(grad_gt, 1))

            print("prediction:", grad_pred.shape)
            print("ground truth:", np.sum(grad_gt, 0))

            y_label.append(grad_gt)
            y_pred.append(grad_pred)


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
    grad_evaluation()