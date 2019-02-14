nohup python3 main.py --gpu_device 2 --depth 1 --num_epochs 80 --fold 1 --lr 1e-5 --reduce_lr_factor 0.5 --img_size 64 > Saliency_depth1.txt &

nohup python3 test_img.py --gpu_device 3 --depth 2 --num_epochs 0 --fold 1 --lr 1e-5 --reduce_lr_factor 0.5 --img_size 64 > test_false_positive.txt &
