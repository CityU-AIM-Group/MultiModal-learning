
### weighted 4teachers, student model with our distillation method.
CUDA_VISIBLE_DEVICES=7 nohup python3 train_cv_path_multi_MT.py --distill crd -a 1 -b 0.05 --nce_p 6 --nce_k 4096 \
        --CE_grads True --niter_decay 30 --fixed_model stage1_ours_colorjit_v1 --reg_type none --beta1 0.9 --pos_mode exact \
        --neg_mode all_others --start_reweight 0 --pos_extra neighbors --model_name random_test --max_discrep 1 \
        --num_teachers 2 --assign_weights True --grads_thresh 0.25 --use_grads_thresh True > stage2_weighted_4teachers.log 2>&1

