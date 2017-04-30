rm ~/gps/data/reacher_color_blocks/*
for i in {0..9} #{0..12}
    do
        python python/gps/gps_main.py --start $i reacher_maml_bc_color_blocks
    done

python python/gps/gps_main.py reacher_maml_bc_color_blocks

for i in {2..20}
do
    rm ~/gps/data/reacher_color_blocks/*
    for j in {0..9} #{0..12}
    do
        python python/gps/gps_main.py --start $((10*($i-1)+$j)) reacher_maml_bc_color_blocks
    done
    python python/gps/gps_main.py --restore $((3000*($i-1)-1)) reacher_maml_bc_color_blocks
done