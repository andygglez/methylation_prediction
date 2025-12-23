



/home/andy-bio/.anaconda/envs/lightning/bin/python lightning.training.py --npz chr19.filtered.npz \
                                                                         --num_workers 2 \
                                                                         --batch_size 8 \
                                                                         --epochs 300 \
                                                                         --accelerator cpu
