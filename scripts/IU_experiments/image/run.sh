#!/bin/bash
# export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=$(pwd):$PYTHONPATH

PRETRAIN_IMAGENET=./pre_checkpoints/vit_b_16-c867db91.pth
DATA_RATIO=0.05

######################################CL baseline 4 tasks#############################################
NUM_FIRST_CLS=80
PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))
TIME=$(date "+%Y%m%d%H%M%S")

# GS-LoRA_pure
for lr in 1e-2; do
    for beta in 0.15; do
        CUDA_VISIBLE_DEVICES=7 python3 -u train/train_own_forget_cl.py -b 48 -w 0 -d imagenet100 -n VIT_B16 -e 100 \
            -head CosFace --outdir ./LOG/test_image100/4types/20250218/type1_lw2g_back_df_nogumble_searchparams_20240216/drcoef-01_lw2g_dr-01_df-005/exps_image/multistep/CLGSLoRA_pure_celoss-dr-coef_1/start${NUM_FIRST_CLS}forgetper${PER_FORGET_CLS}lr${lr}beta${beta}-${TIME} \
            --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs 100 \
            --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
            -r $PRETRAIN_IMAGENET --data_ratio $DATA_RATIO \
            --BND 105 --beta $beta --alpha 0 --min-lr 1e-5 --num_tasks 4 --wandb_group forget_cl_new \
            --cl_beta_list 0.2 0.25 0.25 0.25   \
            --pro_f_weight 0.01 --ema_epoch 30 --ema_decay 0.9 --cl_prof_list 0.01 0.01 0.01 0.01 --method_name FG-OrIU \
            --lw2g \
            --config_name type1_lw2g_back_df_nogumble \
            --ce_loss_on_Dr_coef 0.1 \
            --forward_disturbing_on_DF_coef 0.05 \
            --forward_denoising_on_DR_coef 0.1 \
            --ce_loss_on_Df 0 \
            --ce_loss_on_Dr 1 &
    done
done
