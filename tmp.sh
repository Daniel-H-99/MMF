#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

### Recon on Varying mode, pool, number and temperature for searching

M_LIST="
O
"

P_LIST="
XL
L
M
S
"

N_LIST="
1
10
30
50
100
"

T_LIST="
0.1
0.3
0.5
1.0
"

for P in $P_LIST
do
    for M in $M_LIST
    do
        for N in $N_LIST
        do
            if [ $M != 'O' -a $N -le 10 ]
            then
                continue
            fi
            for T in $T_LIST
            do
                conda activate fom
                echo 'P : '${P}' M: '${M}' N: '${N}' T: '${T}
                python mesh_search.py --lipdisc_path expert_v3/00010000.pt --data_dir ../datasets/kkj_v2/test/studio_1_6.mp4 --pool_dir ../datasets/kkj_v2/pool_$P --mode $M --N $N --T $T --device_id 2
                python mesh_demo.py --config config/kkj-256.yaml --data_dir ../datasets/kkj_v2/test/studio_1_6.mp4 --checkpoint kkj_v2/v1.0/00000309-checkpoint.pth.tar --result_video ${P}_${M}_${N}_${T}_recon.mp4 --device_id 2
                conda deactivate
                conda activate a2l
                python ../ObamaData/util/paste_patch.py --patch_dir demo_img --data_dir ../datasets/kkj_v2/test/studio_1_6.mp4 --resize 360 --name ${P}_${M}_${N}_${T}_pasted
                conda deactivate
            done
        done
    done
done


### Compare Result
dir=../datasets/kkj_v2/test/studio_1_6.mp4
ffmpeg -i $dir/XL_O_1_0.1_pasted.mp4 -i $dir/XL_O_10_0.1_pasted.mp4 -i $dir/XL_O_20_0.1_pasted.mp4 -i $dir/XL_O_50_0.1_pasted.mp4 -i $dir/XL_O_100_0.1_pasted.mp4  -filter_complex hstack=inputs=5 $dir/varing_N.mp4
ffmpeg -i $dir/XL_O_30_0.1_pasted.mp4 -i $dir/XL_O_30_0.3_pasted.mp4 -i $dir/XL_O_30_0.5_recon.mp4 -i $dir/XL_O_30_1.0_pasted.mp4 -filter_complex hstack=inputs=4 $dir/varing_T.mp4
ffmpeg -i $dir/S_O_30_0.3_pasted.mp4 -i $dir/M_O_30_0.3_pasted.mp4 -i $dir/L_O_30_0.3_pasted.mp4 -i $dir/XL_O_30_0.3_pasted.mp4 -filter_complex hstack=inputs=4 $dir/varing_P.mp4
ffmpeg -i $dir/XL_A_30_0.3_pasted.mp4 -i $dir/XL_L_30_0.3_pasted.mp4 -i $dir/XL_O_30_0.3_pasted.mp4 -filter_complex hstack=inputs=3 $dir/varing_M.mp4


# ### Predictor


### Expert
# T_LIST="
# 5
# 3
# "
# for T in $T_LIST
# do
#     python mesh_train_expert.py --ckpt_dir expert_v3.1_W$T --window $T
# done

# W_LIST="
# 0.2
# 0.4
# 1.0
# "

# for W in $W_LIST
# do
#     python mesh_train_vocoder.py --lipdisc_weight $W --ckpt_dir lw_$W
#     # python mesh_test_vocoder.py --data_dir ../datasets/kkj_v2/test/studio_1_6.mp4 --ckpt_path vocoder/lw_$W/best.pt --result_dir vocoder_lw_$W
#     # python mesh_estimate_from_lip.py --config config/kkj-256.yaml --checkpoint log/v8.4/00000099-checkpoint.pth.tar
#     # python python mesh_demo.py --config config/kkj-256.yaml --checkpoint log/v8.4/00000099-checkpoint.pth.tar
# done

