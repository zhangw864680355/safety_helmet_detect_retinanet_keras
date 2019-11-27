#! /bin/bash
retinanet-train --batch-size 2 --steps 48059 --epochs 50 --multi-gpu 2 --multi-gpu-force  csv ./datasets/voc_train_annotation.csv ./datasets/classes.csv --val-annotations ./datasets/voc_val_annotation.csv
