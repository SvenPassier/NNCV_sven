wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --resume-checkpoint "checkpoints/pretrained/model.pth" \
    --batch-size 2 \
    --accumulation_steps 8 \
    --epochs 10 \
    --lr 0.0001 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "model4train4" \


### OLD CODE

# wandb login

# python3 train.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.001 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "unet-training" \