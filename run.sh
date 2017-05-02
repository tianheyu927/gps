# rm ~/gps/data/reacher_color_blocks/*
for i in {3..9} #{0..12}
    do
        python python/gps/gps_main.py --start $(($i+20)) reacher_maml_bc_color_blocks
    done

python python/gps/gps_main.py --restore 5999 --maml_idx 2 reacher_maml_bc_color_blocks

for i in {4..10}
do
    rm ~/gps/data/reacher_color_blocks/*
    for j in {0..9} #{0..12}
    do
        python python/gps/gps_main.py --start $((10*($i-1)+$j)) reacher_maml_bc_color_blocks
    done
    python python/gps/gps_main.py --restore $((3000*($i-1)-1)) --maml_idx $(($i-1)) reacher_maml_bc_color_blocks
done