CUDA_VISIBLE_DEVICES=0 python3 train_cv_MT_SP_Masking.py --pred_distill 1 --CRD_distill 0 --masking 1 --fusion_type pofusion \
    --Path_K 1 --Omic_K 5 --start_epoch 1 --model_name stage1_masked_teacher > 20230818_stage1_masked_teacher.log 2>&1 &

