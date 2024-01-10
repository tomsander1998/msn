python logistic_eval.py \
  --subset-path imagenet_subsets1/1percent.txt \
  --root-path /datasets01/ \
  --image-folder imagenet_full_size/061417/ \
  --device cuda:0 \
  --pretrained "/checkpoint/tomsander/experiments/M3AE/231221_ViP_MAE/"\
  --fname "checkpoint-2.pth" \
  --model-name "vit_base_patch16" \
  --model "model"\
  --penalty l2 \
  --lambd 0.0025

python logistic_eval.py \
  --subset-path imagenet_subsets1/1imgs_class.txt \
  --root-path /datasets01/ \
  --image-folder imagenet_full_size/061417/ \
  --device cuda:0 \
  --pretrained "/checkpoint/tomsander/experiments/M3AE/231221_ViP_MAE/"\
  --fname "checkpoint-2.pth" \
  --model-name "vit_base_patch16" \
  --model "model"\
  --penalty l2 \
  --lambd 0.0025


python logistic_eval.py \
  --subset-path imagenet_subsets1/2imgs_class.txt \
  --root-path /datasets01/ \
  --image-folder imagenet_full_size/061417/ \
  --device cuda:0 \
  --pretrained "/checkpoint/tomsander/experiments/M3AE/231221_ViP_MAE/"\
  --fname "checkpoint-2.pth" \
  --model-name "vit_base_patch16" \
  --model "model"\
  --penalty l2 \
  --lambd 0.0025



python logistic_eval.py \
  --subset-path imagenet_subsets1/5imgs_class.txt \
  --root-path /datasets01/ \
  --image-folder imagenet_full_size/061417/ \
  --device cuda:0 \
  --pretrained "/checkpoint/tomsander/experiments/M3AE/231221_ViP_97k/"\
  --fname "checkpoint-target_step-2.pth" \
  --model-name "vit_base_patch16" \
  --model "model"\
  --penalty l2 \
  --lambd 0.0025


  # --pretrained "/checkpoint/tomsander/experiments/M3AE/231115_eval_vip_40k/"\
  # --fname "checkpoint-target_step-0.pth" \

  # --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_eval_random_vip/"\
  # --fname "checkpoint-6000.pth" \

  # --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_eval_syninit_vip/"\
  # --fname "checkpoint-28-begin.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231115_eval_vip_80k/"\
  # --fname "checkpoint-target_step-1.pth" \

  # --pretrained "/checkpoint/tomsander/experiments/M3AE/231115_eval_vip_160k/"\
  # --fname "checkpoint-target_step-3.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231115_eval_vip_full/"\
  # --fname "checkpoint-target_step-29.pth" \



# python logistic_eval.py \
#   --subset-path imagenet_subsets1/1percent.txt \
#   --root-path /datasets01/ \
#   --image-folder imagenet_full_size/061417/ \
#   --device cuda:0 \
#   --pretrained "/checkpoint/tomsander/experiments/M3AE/231212_Cap97k_8/"\
#   --fname "checkpoint-target_step-2.pth" \
#   --model-name "mae_vit_base_patch16_autoregressive_nobias" \
#   --model "model"\
#   --penalty l2 \
#   --lambd 0.0025

# python logistic_eval.py \
#   --subset-path imagenet_subsets1/1imgs_class.txt \
#   --root-path /datasets01/ \
#   --image-folder imagenet_full_size/061417/ \
#   --device cuda:0 \
#   --pretrained "/checkpoint/tomsander/experiments/M3AE/231221_CAP_400k/"\
#   --fname "checkpoint-target_step-9.pth" \
#   --model-name "mae_vit_base_patch16_autoregressive_nobias" \
#   --model "model"\
#   --penalty l2 \
#   --lambd 0.0025

# python logistic_eval.py \
#   --subset-path imagenet_subsets1/2imgs_class.txt \
#   --root-path /datasets01/ \
#   --image-folder imagenet_full_size/061417/ \
#   --device cuda:0 \
#   --pretrained "/checkpoint/tomsander/experiments/M3AE/231221_CAP_400k/"\
#   --fname "checkpoint-target_step-9.pth" \
#   --model-name "mae_vit_base_patch16_autoregressive_nobias" \
#   --model "model"\
#   --penalty l2 \
#   --lambd 0.0025

# python logistic_eval.py \
#   --subset-path imagenet_subsets1/5imgs_class.txt \
#   --root-path /datasets01/ \
#   --image-folder imagenet_full_size/061417/ \
#   --device cuda:0 \
#   --pretrained "/checkpoint/tomsander/experiments/M3AE/231221_CAP_400k/"\
#   --fname "checkpoint-target_step-9.pth" \
#   --model-name "mae_vit_base_patch16_autoregressive_nobias" \
#   --model "model"\
#   --penalty l2 \
#   --lambd 0.0025

  # --pretrained "/checkpoint/tomsander/experiments/M3AE/231212_Cap200k_next/"\
  # --fname "checkpoint-target_step-4.pth" \

  # --pretrained "/checkpoint/tomsander/experiments/M3AE/231212_Cap97k_8/"\
  # --fname "checkpoint-target_step-2.pth" \

# --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_eval_eps2/"\
# --fname "checkpoint-15-begin.pth" \

  # --pretrained "/checkpoint/tomsander/experiments/M3AE/311023_eps1_true_5e-8_followup/"\
  # --fname "checkpoint-target_step-3.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_eval_epsinf/"\
  # --fname "checkpoint-8-begin.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_eval_rndam_cap/"\
  # --fname "checkpoint-target_step-30.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_eval_small_cap/"\
  # --fname "checkpoint-target_step-30.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_eval_tiny_cap/"\
  # --fname "checkpoint-target_step-30.pth" \
  # --model-name "mae_vit_tiny_patch16_autoregressive_nobias" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231106_TAN_40K/"\
  # --fname "checkpoint-target_step-0.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231106_TAN_80K/"\
  # --fname "checkpoint-target_step-1.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231106_TAN_160K/"\
  # --fname "checkpoint-3.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231116_01_next3/"\
  # --fname "checkpoint-30-begin.pth" \

  #   --pretrained "/checkpoint/tomsander/experiments/M3AE/231114_DPCAP90k_3/"\
  # --fname "checkpoint-1-begin.pth" \