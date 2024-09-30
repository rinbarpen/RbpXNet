python main.py --project 'Segment-Drive' \
    -m "UNet" --n_channels 1 --n_classes 1 \
    --classes "vein" \
    -lr 3e-5 -e 40 -b 1 \
    --data_dir '/media/tangh/新加卷/Data/Datasets/' \
    --dataset DRIVE \
    --gpu \
    --augment_boost \
    --train \
    --save_every_n_epoch 1
