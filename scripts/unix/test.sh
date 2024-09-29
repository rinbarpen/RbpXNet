python main.py --project 'Segment-Drive' \
    -m "UNet" --n_channels 1 --n_classes 1 --classes "vein" \
    -b 1 \
    --data_dir '/media/tangh/新加卷/Data/Datasets/' --dataset 'DRIVE' \
    --gpu \
    --test
