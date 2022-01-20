#!/bin/bash


# echo 'working on train data'
# for file in ../datasets/kkj_v2/train/*.mp4
# do
#     python pose_normalization.py --data_dir $file --draw_mesh --interpolate_z --ref_path ../datasets/train_kkj/kkj04.mp4/frame_reference.png
# done

# python -c 'from modules.util import construct_mesh; construct_mesh("../datasets/kkj_v2/train");'
# for file in ../datasets/kkj_v2/train/*.mp4
# do
#     python -c 'from modules.util import project_mesh; project_mesh(${file}, "../datasets/train_kkj/kkj04.mp4/mesh_pca.pt", "../mesh_dict_reference.pt")'
# done

# echo 'working on test data'
# for file in ../datasets/kkj_v2/test/*.mp4
# do
#     python pose_normalization.py --data_dir $file --draw_mesh --interpolate_z --ref_path ../datasets/train_kkj/kkj04.mp4/frame_reference.png
# done

# python -c 'from modules.util import construct_mesh; construct_mesh("../datasets/kkj_v2/test");'
# for file in ../datasets/kkj_v2/test/*.mp4
# do
#     python -c 'from modules.util import project_mesh; project_mesh(${file}, "../datasets/train_kkj/kkj04.mp4/mesh_pca.pt", "../mesh_dict_reference.pt")'
# done

for file in ../data_preprocessed/lof/*.mp4
do
    python pose_normalization.py --data_dir $file --draw_mesh --interpolate_z --ref_path ../datasets/train_kkj/kkj04.mp4/frame_reference.png
done

# python -c 'from modules.util import construct_mesh; construct_mesh("../datasets/data_preprocessed/lof");'

# for file in ../datasets/data_preprocessed/lof/*.mp4
# do
#     python -c 'from modules.util import project_mesh; project_mesh('${file}', "../datasets/train_kkj/kkj04.mp4/mesh_pca.pt", "../mesh_dict_reference.pt")'
# done

# construct search pool
# XS: 1 video S: 5000 M: 10000 L: 20000 XL: all
# DATA_DIR=../datasets/kkj_v2/train

# LABEL_LIST=(XS S M L XL)

# N_LIST=(1000 5000 10000 20000)

# for i in $(seq 0 4)
# do
#     LABEL=${LABEL_LIST[i]}
#     if [ $i -eq 4 ]
#     then 
#         python -c 'from modules.util import construct_pool; construct_pool("'$DATA_DIR'", pool_name="../pool_'${LABEL}'")'
#     else
#         N=${N_LIST[i]}
#         python -c 'from modules.util import construct_pool; construct_pool("'$DATA_DIR'", N='${N}', pool_name="../pool_'${LABEL}'")'
#     fi
# done


