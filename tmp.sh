tgt_dir='../datasets/etc/son.mp4'
src_dir='../datasets/etc/kmj.mp4'
# ffmpeg -framerate 25 -start_number 1 -i $tgt_dir/img/%05d.png -vframes 33 $tgt_dir/chunk_1.mp4
# ffmpeg -framerate 25 -start_number 50 -i $tgt_dir/img/%05d.png -vframes 42 $tgt_dir/chunk_2.mp4
# ffmpeg -framerate 25 -start_number 399 -i $tgt_dir/img/%05d.png -vframes 23 $tgt_dir/chunk_3.mp4
# ffmpeg -framerate 25 -start_number 478 -i $tgt_dir/img/%05d.png -vframes 44 $tgt_dir/chunk_4.mp4

python demo.py --config config/vox-256.yaml --checkpoint log/vox-cpk.pth.tar --driving_video $tgt_dir/chunk_1.mp4 --source_image $src_dir/crop/00030.png --result_video $tgt_dir/kmj_chunk_1.mp4 --relative --adapt_scale
python demo.py --config config/vox-256.yaml --checkpoint log/vox-cpk.pth.tar --driving_video $tgt_dir/chunk_2.mp4 --source_image $src_dir/crop/00030.png --result_video $tgt_dir/kmj_chunk_2.mp4 --relative --adapt_scale
python demo.py --config config/vox-256.yaml --checkpoint log/vox-cpk.pth.tar --driving_video $tgt_dir/chunk_3.mp4 --source_image $src_dir/crop/00030.png --result_video $tgt_dir/kmj_chunk_3.mp4 --relative --adapt_scale
python demo.py --config config/vox-256.yaml --checkpoint log/vox-cpk.pth.tar --driving_video $tgt_dir/chunk_4.mp4 --source_image $src_dir/crop/00030.png --result_video $tgt_dir/kmj_chunk_4.mp4 --relative --adapt_scale
