python main.py --project "Segment-Drive" \
    -m "UNet" \
    --n_channels 1 --n_classes 1 --classes "vein" \
    --gpu \
    --predict \
    --load 'output/best_model.pth' \
    --input "/media/tangh/新加卷/Data/Datasets/DRIVE/test/images" 
