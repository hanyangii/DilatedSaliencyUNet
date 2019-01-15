#nohup python3 finetune_UNet.py --gpu_device 2 --depth 3 --num_epochs 100 --fold 1 --lr 1e-6 --init_num 20 --reduce_lr_factor 0.5 --dir_name LecuninitEncode20_lr1e-6 --finetuning > nohup_out_txt/Finetune_LecuninitEncode20_depth3_lr1e-6_crossentropy.txt &

nohup python3 test_img.py --gpu_device 2 --depth 3 --num_epochs 0 --fold 1 --lr 1e-5 --reduce_lr_factor 0.5 --img_size 64 --dir_name vis --restore > test_time.txt &
