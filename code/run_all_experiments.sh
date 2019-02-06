# !/bin/sh
python -u train_segroot.py --width 2 > "log_SegRoot(2,5).txt"
python -u train_segroot.py --width 16 --depth 4 --lr 1e-3 > "log_SegRoot(16,4).txt"
python -u train_segroot.py --width 32 --depth 5 --lr 1e-4 --bs 32 > "log_SegRoot(32,5).txt"
python -u train_segroot.py --width 64 --depth 4 --lr 1e-4 --bs 16 > "log_SegRoot(64,4).txt"
python -u train_segroot.py --width 64 --depth 5 --lr 2e-5 --bs 8 --epochs 100 --verbose 2 > "log_SegRoot(64,5).txt"
