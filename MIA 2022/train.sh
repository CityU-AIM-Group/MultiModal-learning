# ### stage1, multi-modal fusion with M3LR constraint.
# python3 train_cv_MT.py --pred_distill 1 --CRD_distill 0 --model_name random_test \
#     --tSVD_loss True --mode pathomic --init_type max --beta1 0.5 --fusion_type pofusion \
#     --n_views 4 --tSVD_mode pathomic --Lambda_global 0.1 --batch_size 16 \
#     --path_dim 128 --omic_dim 128 --mmhid 128 ;


# ### stage1, multi-modal fusion with traditional low-rank constraint（只在两个单模态之间约束low-rank）.
# python3 train_cv_MT.py --pred_distill 0 --CRD_distill 0 --model_name 1217_pofusion_t-SVD-MSC \
#     --tSVD_loss True --mode pathomic --init_type max --beta1 0.5 --fusion_type pofusion \
#     --n_views 2 --tSVD_mode pathomic --Lambda_global 0.05 ;


# ### stage1, multi-modal fusion with M3LR constraint.
# python3 train_cv_MT.py --pred_distill 0 --CRD_distill 0 --model_name 1216_pofusion_baseline_bs16_dim64 \
#     --tSVD_loss False --mode pathomic --init_type max --beta1 0.5 --fusion_type pofusion \
#     --batch_size 16 --path_dim 64 --omic_dim 64 --mmhid 64 ;


# ### stage1, test the model.
# python3 test_cv_MT.py --mode pathomic --model_name 1216_pofusion_baseline_bs16_dim128 --fusion_type pofusion \
#     --path_dim 128 --omic_dim 128 --mmhid 128 ;


### path branch, KL div + CRD (exact pos, 1024 neg)
python3 train_cv_path_multi_MT.py --distill crd -r 1.0 -a 1.0 -b 0.02 --CE_grads True --niter_decay 30 \
      --model_name random_test --fixed_model 1023_pathomic_MT --reg_type none --beta1 0.9 --kd_T 1.0 \
      --nce_p 1 --nce_p2 1 --nce_k 1024 --nce_k2 1024 --neg_reweight False --pos_mode exact ;


# ### path branch, KL div + DSCD (hard 20 pos, 1024 neg)
# python3 train_cv_path_multi_MT.py --distill crd -r 1.0 -a 1.0 -b 0.02 --CE_grads True --niter_decay 30 \
#       --model_name random_test --reg_type none --beta1 0.9 --kd_T 1.0 \
#       --select_pos_mode hard --nce_p2 20 --nce_k 1024 --nce_k2 1024 --neg_reweight False ;


# ### path branch, 4 teachers, grads_momentum = 0.5, grads_thresh = 0.1.
# python3 train_cv_path_multi_MT.py --distill crd -a 1 -b 0.02 --nce_p2 20 --num_teachers 2 \
#       --CE_grads True --model_name 1217_path_weighted_4teachers_mo_0.5_thresh_0.1 \
#       --reg_type none --beta1 0.9 --kd_T 1.0 --select_pos_mode hard --assign_weights True \
#       --nce_k 1024 --nce_k2 1024 ;
