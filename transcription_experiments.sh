#!/bin/bash

epoches=2000
seed=3281514

function evaluate_model {
    python evaluate_eps.py with weight_file=$1 dataset=GuitarSet device=cuda:1
    python evaluate.py with weight_file=$1 dataset=GuitarSet device=cuda:1

    python evaluate_eps.py with weight_file=$1 dataset=MAPS device=cuda:1
    python evaluate.py with weight_file=$1 dataset=MAPS device=cuda:1
    
    python evaluate.py with weight_file=$1 dataset=SynthesizedInstruments device=cuda:1
}

# function evaluate_model {
#     python evaluate_eps.py with weight_file=$1 dataset=GuitarSet device=cuda:1
#     python evaluate_eps.py with weight_file=$1 dataset=MAPS device=cuda:1
# }

# function evaluate_model {
#     python evaluate.py with weight_file=$1 dataset=GuitarSet device=cuda:1
#     python evaluate.py with weight_file=$1 dataset=MAPS device=cuda:1
#     python evaluate.py with weight_file=$1 dataset=SynthesizedInstruments device=cuda:1
# }

# #BASIC TRANSCRIPTION
# python train.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS model_type=unet epoches=$epoches seed=$seed
# python train.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet model_type=unet epoches=$epoches seed=$seed
# python train.py with train_on=SynthesizedInstruments logdir=results/unet_model_trained_on_SynthesizedInstruments model_type=unet epoches=$epoches seed=$seed

# #EVALUATE TRANSCRIPTION
# evaluate_model results/unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt

# #TRANSFER FROM SYNTHESIZED INSTRUMENTS
# python train.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed
# python train.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed

# #EVALUATE TRANSFER
# evaluate_model results/transferred_unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet/model-$epoches.pt

# #TRANSFER FROM GUITARSET
# python train.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS/model-$epoches.pt

# #TRANSFER FROM MAPS
# python train.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet/model-$epoches.pt

# python result_dict_analysis.py

# python result_table_generator.py results/unet_model_trained_on_MAPS results/unet_model_trained_on_GuitarSet results/unet_model_trained_on_SynthesizedInstruments  results/transferred_unet_model_trained_on_MAPS results/transferred_unet_model_trained_on_GuitarSet results/transferred_from_guitarset_unet_model_trained_on_MAPS results/transferred_from_MAPS_unet_model_trained_on_GuitarSet > results/table.txt

# evaluate_model results/unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet/model-$epoches.pt

python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.1 model_type=unet epoches=4000 seed=$seed train_size=.1
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.2 model_type=unet epoches=3000 seed=$seed train_size=.2
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.3 model_type=unet epoches=$epoches seed=$seed train_size=.3
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.4 model_type=unet epoches=$epoches seed=$seed train_size=.4
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.5 model_type=unet epoches=$epoches seed=$seed train_size=.5
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.6 model_type=unet epoches=$epoches seed=$seed train_size=.6
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.7 model_type=unet epoches=$epoches seed=$seed train_size=.7
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.8 model_type=unet epoches=$epoches seed=$seed train_size=.8
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.9 model_type=unet epoches=$epoches seed=$seed train_size=.9

# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.01 model_type=unet epoches=4000 seed=$seed train_size=.01
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.02 model_type=unet epoches=4000 seed=$seed train_size=.02
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.05 model_type=unet epoches=4000 seed=$seed train_size=.05
# python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.15 model_type=unet epoches=3000 seed=$seed train_size=.15

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.1_$i model_type=unet epoches=4000 seed=328151$i train_size=.1
# done

# for i in $(seq 1 7);
# do
#     python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.2_$i model_type=unet epoches=3000 seed=328151$i train_size=.2
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.005_$i model_type=unet epoches=4000 seed=328151$i train_size=.005
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.01_$i model_type=unet epoches=4000 seed=328151$i train_size=.01
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.02_$i model_type=unet epoches=4000 seed=328151$i train_size=.02
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.05_$i model_type=unet epoches=4000 seed=328151$i train_size=.05
# done

# for i in $(seq 1 7);
# do
#     python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.15_$i model_type=unet epoches=3000 seed=328151$i train_size=.15
# done

python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.1 model_type=unet epoches=4000 seed=$seed train_size=.1
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.2 model_type=unet epoches=3000 seed=$seed train_size=.2
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.3 model_type=unet epoches=$epoches seed=$seed train_size=.3
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.4 model_type=unet epoches=$epoches seed=$seed train_size=.4
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.5 model_type=unet epoches=$epoches seed=$seed train_size=.5
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.6 model_type=unet epoches=$epoches seed=$seed train_size=.6
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.7 model_type=unet epoches=$epoches seed=$seed train_size=.7
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.8 model_type=unet epoches=$epoches seed=$seed train_size=.8
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.9 model_type=unet epoches=$epoches seed=$seed train_size=.9

# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.01 model_type=unet epoches=4000 seed=$seed train_size=.01
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.02 model_type=unet epoches=4000 seed=$seed train_size=.02
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.05 model_type=unet epoches=4000 seed=$seed train_size=.05
# python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.15 model_type=unet epoches=4000 seed=$seed train_size=.15

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.1_$i model_type=unet epoches=4000 seed=328151$i train_size=.1
# done

# for i in $(seq 1 7);
# do
#     python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.2_$i model_type=unet epoches=3000 seed=328151$i train_size=.2
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.005_$i model_type=unet epoches=4000 seed=328151$i train_size=.005
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.01_$i model_type=unet epoches=4000 seed=328151$i train_size=.01
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.02_$i model_type=unet epoches=4000 seed=328151$i train_size=.02
# done

# for i in $(seq 1 9);
# do
#     python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.05_$i model_type=unet epoches=4000 seed=328151$i train_size=.05
# done

# for i in $(seq 1 7);
# do
#     python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.15_$i model_type=unet epoches=3000 seed=328151$i train_size=.15
# done

python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.1 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=4000 seed=$seed train_size=.1
python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.2 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=3000 seed=$seed train_size=.2
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.3 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed train_size=.3
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.4 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed train_size=.4
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.5 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed train_size=.5
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.6 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed train_size=.6
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.7 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed train_size=.7
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.8 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed train_size=.8
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.9 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed train_size=.9

# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.005 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=4000 seed=$seed train_size=.005
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.01 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=4000 seed=$seed train_size=.01
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.02 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=4000 seed=$seed train_size=.02
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.05 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=4000 seed=$seed train_size=.05
# python train_reduced.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS_.15 model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=3000 seed=$seed train_size=.15

python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.1 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=4000 seed=$seed train_size=.1
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.2 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=3000 seed=$seed train_size=.2
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.3 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.3
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.4 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.4
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.5 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.5
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.6 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.6
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.7 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.7
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.8 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.8
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.9 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.9

# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.005 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=4000 seed=$seed train_size=.005
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.01 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=4000 seed=$seed train_size=.01
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.02 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=4000 seed=$seed train_size=.02
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.05 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=4000 seed=$seed train_size=.05
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.15 model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=3000 seed=$seed train_size=.15

python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.1 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.1
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.2 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=3000 seed=$seed train_size=.2
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.3 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.3
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.4 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.4
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.5 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.5
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.6 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.6
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.7 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.7
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.8 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.8
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.9 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.9

# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.005 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.005
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.01 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.01
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.02 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.02
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.05 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.05
# python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.15 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=3000 seed=$seed train_size=.15

python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.1 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.1
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.2 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=3000 seed=$seed train_size=.2
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.3 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.3
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.4 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.4
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.5 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.5
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.6 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.6
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.7 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.7
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.8 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.8
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.9 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.9

# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.005 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.005
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.01 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.01
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.02 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.02
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.05 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=4000 seed=$seed train_size=.05
# python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.15 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=3000 seed=$seed train_size=.15

evaluate_model results/unet_model_trained_on_MAPS_.1/model-4000.pt
evaluate_model results/unet_model_trained_on_MAPS_.2/model-3000.pt
# evaluate_model results/unet_model_trained_on_MAPS_.3/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_MAPS_.4/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_MAPS_.5/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_MAPS_.6/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_MAPS_.7/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_MAPS_.8/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_MAPS_.9/model-$epoches.

# evaluate_model results/unet_model_trained_on_MAPS_.005/model-4000.pt
# evaluate_model results/unet_model_trained_on_MAPS_.01/model-4000.pt
# evaluate_model results/unet_model_trained_on_MAPS_.02/model-4000.pt
# evaluate_model results/unet_model_trained_on_MAPS_.05/model-4000.pt
# evaluate_model results/unet_model_trained_on_MAPS_.15/model-3000.pt

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_MAPS_.1_$i/model-4000.pt
# done

# for i in $(seq 1 7);
# do
#     evaluate_model results/unet_model_trained_on_MAPS_.2_$i/model-3000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_MAPS_.005_$i/model-4000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_MAPS_.01_$i/model-4000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_MAPS_.02_$i/model-4000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_MAPS_.05_$i/model-4000.pt
# done

# for i in $(seq 1 7);
# do
#     evaluate_model results/unet_model_trained_on_MAPS_.15_$i/model-3000.pt
# done

# evaluate_model results/unet_model_trained_on_GuitarSet_.1/model-4000.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.2/model-3000.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.3/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.4/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.5/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.6/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.7/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.8/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.9/model-$epoches.pt

# evaluate_model results/unet_model_trained_on_GuitarSet_.005/model-4000.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.01/model-4000.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.02/model-4000.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.05/model-4000.pt
# evaluate_model results/unet_model_trained_on_GuitarSet_.15/model-3000.pt

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_GuitarSet_.1_$i/model-4000.pt
# done

# for i in $(seq 1 7);
# do
#     evaluate_model results/unet_model_trained_on_GuitarSet_.2_$i/model-3000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_GuitarSet_.005_$i/model-4000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_GuitarSet_.01_$i/model-4000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_GuitarSet_.02_$i/model-4000.pt
# done

# for i in $(seq 1 9);
# do
#     evaluate_model results/unet_model_trained_on_GuitarSet_.05_$i/model-4000.pt
# done

# for i in $(seq 1 7);
# do
#     evaluate_model results/unet_model_trained_on_GuitarSet_.15_$i/model-3000.pt
# done

evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.1/model-4000.pt
evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.2/model-3000.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.3/model-$epoches.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.4/model-$epoches.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.5/model-$epoches.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.6/model-$epoches.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.7/model-$epoches.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.8/model-$epoches.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.9/model-$epoches.pt

# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.005/model-4000.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.01/model-4000.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.02/model-4000.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.05/model-4000.pt
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS_.15/model-3000.pt

evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.1/model-4000.pt
evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.2/model-3000.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.3/model-$epoches.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.4/model-$epoches.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.5/model-$epoches.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.6/model-$epoches.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.7/model-$epoches.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.8/model-$epoches.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.9/model-$epoches.pt

# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.005/model-4000.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.01/model-4000.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.02/model-4000.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.05/model-4000.pt
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet_.15/model-3000.pt

evaluate_model results/transferred_unet_model_trained_on_MAPS_.1/model-4000.pt
evaluate_model results/transferred_unet_model_trained_on_MAPS_.2/model-3000.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.3/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.4/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.5/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.6/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.7/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.8/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.9/model-$epoches.

# evaluate_model results/transferred_unet_model_trained_on_MAPS_.005/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.01/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.02/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.05/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_MAPS_.15/model-3000.pt

evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.1/model-4000.pt
evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.2/model-3000.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.3/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.4/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.5/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.6/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.7/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.8/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.9/model-$epoches.pt

# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.005/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.01/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.02/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.05/model-4000.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet_.15/model-3000.pt

function evaluate_model {
    python evaluate_eps.py with weight_file=$1 dataset=MAPS device=cuda:1
    python evaluate.py with weight_file=$1 dataset=MAPS device=cuda:1
    
    python evaluate_eps.py with weight_file=$1 dataset=MAESTRO device=cuda:1
    python evaluate.py with weight_file=$1 dataset=MAESTRO device=cuda:1
    
    python evaluate.py with weight_file=$1 dataset=SynthesizedInstruments device=cuda:1
}

# python train.py with train_on=MAESTRO logdir=results_optional/unet_model_trained_on_MAESTRO model_type=unet epoches=$epoches seed=$seed
# python train.py with train_on=MAESTRO logdir=results_optional/transferred_from_MAPS_unet_model_trained_on_MAESTRO model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed
# python train.py with train_on=MAPS logdir=results_optional/transferred_from_MAESTRO_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_MAESTRO/model-$epoches.pt epoches=$epoches seed=$seed
# python train.py with train_on=MAESTRO logdir=results_optional/transferred_unet_model_trained_on_MAESTRO model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed

# evaluate_model results_optional/unet_model_trained_on_MAESTRO/model-$epoches.pt
# evaluate_model results_optional/transferred_from_MAPS_unet_model_trained_on_MAESTRO/model-$epoches.pt
# evaluate_model results_optional/transferred_from_MAESTRO_unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results_optional/transferred_unet_model_trained_on_MAESTRO/model-$epoches.pt

# python train_reduced.py with train_on=MAESTRO logdir=results_optional/unet_model_trained_on_MAESTRO_.23 model_type=unet epoches=$epoches seed=$seed train_size=.23
# python train_reduced.py with train_on=MAESTRO logdir=results_optional/transferred_from_MAPS_unet_model_trained_on_MAESTRO_.23 model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed train_size=.23
# python train_reduced.py with train_on=MAESTRO logdir=results_optional/transferred_unet_model_trained_on_MAESTRO_.23 model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.23
# python train.py with train_on=MAPS logdir=results_optional/transferred_from_MAESTRO_.23_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_MAESTRO_.23/model-$epoches.pt epoches=$epoches seed=$seed

# evaluate_model results_optional/unet_model_trained_on_MAESTRO_.23/model-$epoches.pt
# evaluate_model results_optional/transferred_from_MAPS_unet_model_trained_on_MAESTRO_.23/model-$epoches.pt
# evaluate_model results_optional/transferred_unet_model_trained_on_MAESTRO_.23/model-$epoches.pt
# evaluate_model results_optional/transferred_from_MAESTRO_.23_unet_model_trained_on_MAPS/model-$epoches.pt

function evaluate_model {
    python evaluate_eps.py with weight_file=$1 dataset=MAESTRO device=cuda:1
    python evaluate.py with weight_file=$1 dataset=MAESTRO device=cuda:1
}

# evaluate_model results_optional/unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results_optional/transferred_unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results_optional/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt

# python train.py with train_on=MAPSSynthesizedInstruments logdir=results_optional/unet_model_trained_on_MAPSSynthesizedInstruments model_type=unet epoches=$epoches seed=$seed
# python train.py with train_on=FullSynthesizedInstruments logdir=results_optional/unet_model_trained_on_FullSynthesizedInstruments model_type=unet epoches=$epoches seed=$seed

# python train.py with train_on=MAPS logdir=results_optional/transferred_from_MAPS_synthesized_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_MAPSSynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed
# python train.py with train_on=GuitarSet logdir=results_optional/transferred_from_MAPS_synthesized_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results_optional/unet_model_trained_on_MAPSSynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed

# evaluate_model results_optional/unet_model_trained_on_FullSynthesizedInstruments/model-$epoches.pt
# evaluate_model results_optional/unet_model_trained_on_MAPSSynthesizedInstruments/model-$epoches.pt
# evaluate_model results_optional/transferred_from_MAPS_synthesized_unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results_optional/transferred_from_MAPS_synthesized_unet_model_trained_on_GuitarSet/model-$epoches.pt