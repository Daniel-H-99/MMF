#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

### Recon on Varying mode, pool, number and temperature for searching

# M_LIST="
# O
# "

# P_LIST="
# XL
# "

# N_LIST="
# 10
# "

# T_LIST="
# 0.5
# "

# W1_LIST="
# 1
# 0.1
# 0.01
# 0.001
# "

# W2_LIST="
# 0
# "

# K_LIST="
# 9
# "

# T=0.5

# for P in $P_LIST
# do
#     for M in $M_LIST
#     do
#         for N in $N_LIST
#         do
#             if [ $M != 'O' -a $N -le 10 ]
#             then
#                 continue
#             fi
#             for K in $K_LIST
#             do
#                 for W1 in $W1_LIST
#                 do
#                     for W2 in $W2_LIST
#                     do
#                         conda activate fom
#                         echo 'P : '${P}' M: '${M}' N: '${N}' T: '${T}' K: '${K}
#                         python mesh_search.py --lipdisc_path expert_v3.1_W5/best.pt --data_dir ../datasets/kkj_v2/test/studio_1_2.mp4 --pool_dir ../datasets/kkj_v2/pool_$P --mode $M --N $N --T $T --device_id 2 --l1_weight $W1 --l2_weight $W2 --k $K
#                         python mesh_estimate_from_lip.py --config config/kkj-256.yaml --checkpoint kkj_v2/v1.0/00000309-checkpoint.pth.tar
#                         python mesh_demo.py --config config/kkj-256.yaml --data_dir ../datasets/kkj_v2/test/studio_1_2.mp4 --checkpoint kkj_v2/v1.0/00000309-checkpoint.pth.tar --result_video ${P}_${M}_${N}_${T}_${W1}_${W2}_${K}_recon.mp4 --device_id 2
#                         conda deactivate
#                         conda activate a2l
#                         python ../ObamaData/util/paste_patch.py --patch_dir demo_img --data_dir ../datasets/kkj_v2/test/studio_1_2.mp4 --resize 360 --name ${P}_${M}_${N}_${T}_${W1}_${W2}_${K}_pasted
#                         conda deactivate
#                     done
#                 done
#             done
#         done
#     done
# done


### Compare Result
# dir=../datasets/kkj_v2/test/studio_1_6.mp4
# ffmpeg -y -i $dir/XL_O_1_0.5_pasted.mp4 -i $dir/XL_O_10_0.5_pasted.mp4 -i $dir/XL_O_30_0.5_pasted.mp4 -i $dir/XL_O_50_0.5_pasted.mp4 -i $dir/XL_O_100_0.5_pasted.mp4  -filter_complex hstack=inputs=5 $dir/varing_N.mp4
# ffmpeg -y -i $dir/XL_O_30_0.1_pasted.mp4 -i $dir/XL_O_30_0.5_pasted.mp4 -i $dir/XL_O_30_1.0_pasted.mp4 -filter_complex hstack=inputs=3 $dir/varing_T.mp4
# ffmpeg -y -i $dir/S_O_30_0.5_pasted.mp4 -i $dir/M_O_30_0.5_pasted.mp4 -i $dir/L_O_30_0.5_pasted.mp4 -i $dir/XL_O_30_0.5_pasted.mp4 -filter_complex hstack=inputs=4 $dir/varing_P.mp4
# ffmpeg -y -i $dir/XL_A_30_0.5_pasted.mp4 -i $dir/XL_L_30_0.5_pasted.mp4 -i $dir/XL_O_30_0.5_pasted.mp4 -filter_complex hstack=inputs=3 $dir/varing_M.mp4


# ### Predictor


## Expert
T_LIST="
5
3
"
for T in $T_LIST
do
    python mesh_train_expert.py --ckpt_dir expert_v3.1_W$T --window $T
done

conda deactivate

W_LIST="
0.5
"

LW_LIST="
0.5
0.75
1.0
"

NW_LIST="
0.2
0.3
0.4
"

for W in $W_LIST
do
    for LW in $LW_LIST
    do
        for NW in $NW_LIST
        do
            conda activate fom
            python mesh_train_vocoder.py --weight $W --lipdisc_weight $LW --ckpt_dir ${W}_${LW}_${NW} --denoiser_path denoiser_noise1e-2/00003000.pt --denoise_weight $NW
            python mesh_test_vocoder.py --data_dir ../datasets/kkj_v2/test/studio_1_34.mp4 --ckpt_path vocoder/${W}_${LW}_${NW}/best.pt --result_dir vocoder_${W}_${LW}_${NW}
            python mesh_estimate_from_lip.py --config config/kkj-256.yaml --checkpoint kkj_v2/v1.0/00000309-checkpoint.pth.tar
            python mesh_demo.py --config config/kkj-256.yaml --checkpoint kkj_v2/v1.0/00000309-checkpoint.pth.tar --result_video pred_${W}_${LW}_${NW}_l1_recon.mp4
            conda deactivate
            conda activate a2l
            python ../ObamaData/util/paste_patch.py --patch_dir demo_img --data_dir ../datasets/kkj_v2/test/studio_1_34.mp4 --resize 360 --name pred_${W}_${LW}_${NW}_l1_pasted
            conda deactivate
        done
    done
done

# N_LIST="
# 1e-3
# 2e-3
# 3e-3
# 4e-3
# "

# for N in $N_LIST
# do
#     python mesh_train_denoiser.py --ckpt_dir denoiser_proj_noise$N --noise $N
# done

# N_LIST="
# 1e-3
# 5e-3
# 1e-2
# "

# for N in $N_LIST
# do
#     python mesh_train_denoiser.py --ckpt_dir denoiser_noise$N --noise $N --device_id 3
# done

