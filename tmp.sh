M_LIST="
A
L
O
"

P_LIST="
XL
L
M
S
"

N_LIST="
20
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
            for T in $T_LIST
            do
                python mesh_search.py --lipdisc_path expert_v3/00010000.pt --pool_dir ../datasets/kkj_v2/pool_$P --mode $M --N $N --T $T
                python mesh_demo.py --config config/kkj-256.yaml --checkpoint log/v8.4/00000099-checkpoint.pth.tar --result_video $P_$M_$N_$T_recon.mp4
            done
        done
    done
done

