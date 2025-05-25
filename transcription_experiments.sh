#!/bin/bash

epoches=2000
seed=3281514

function evaluate_model {
    python evaluate.py with weight_file=$1 dataset=GuitarSet
    python evaluate.py with weight_file=$1 dataset=MAPS
    python evaluate.py with weight_file=$1 dataset=SynthesizedInstruments
}

function evaluate_model {
    python evaluate_eps.py with weight_file=$1 dataset=GuitarSet device=cuda:1
    python evaluate_eps.py with weight_file=$1 dataset=MAPS device=cuda:1
}

#BASIC TRANSCRIPTION
python train.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS model_type=unet epoches=$epoches seed=$seed
python train.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet model_type=unet epoches=$epoches seed=$seed
python train.py with train_on=SynthesizedInstruments logdir=results/unet_model_trained_on_SynthesizedInstruments model_type=unet epoches=$epoches seed=$seed

# #EVALUATE TRANSCRIPTION
# evaluate_model results/unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_GuitarSet/model-$epoches.pt
# evaluate_model results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt

#TRANSFER FROM SYNTHESIZED INSTRUMENTS !!! SEED=34 !!!
python train.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed
python train.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed

# #EVALUATE TRANSFER
# evaluate_model results/transferred_unet_model_trained_on_MAPS/model-$epoches.pt
# evaluate_model results/transferred_unet_model_trained_on_GuitarSet/model-$epoches.pt

#TRANSFER FROM GUITARSET
python train.py with train_on=MAPS logdir=results/transferred_from_guitarset_unet_model_trained_on_MAPS model_type=unet pretrained_model_path=results/unet_model_trained_on_GuitarSet/model-$epoches.pt epoches=$epoches seed=$seed
# evaluate_model results/transferred_from_guitarset_unet_model_trained_on_MAPS/model-$epoches.pt

#TRANSFER FROM MAPS
python train.py with train_on=GuitarSet logdir=results/transferred_from_MAPS_unet_model_trained_on_GuitarSet model_type=unet pretrained_model_path=results/unet_model_trained_on_MAPS/model-$epoches.pt epoches=$epoches seed=$seed
# evaluate_model results/transferred_from_MAPS_unet_model_trained_on_GuitarSet/model-$epoches.pt

# python result_dict_analysis.py

# python result_table_generator.py results/unet_model_trained_on_MAPS results/unet_model_trained_on_GuitarSet results/unet_model_trained_on_SynthesizedInstruments  results/transferred_unet_model_trained_on_MAPS results/transferred_unet_model_trained_on_GuitarSet results/transferred_from_guitarset_unet_model_trained_on_MAPS results/transferred_from_MAPS_unet_model_trained_on_GuitarSet > results/table.txt

python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.1 model_type=unet epoches=$epoches seed=$seed train_size=.1
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.2 model_type=unet epoches=$epoches seed=$seed train_size=.2
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.3 model_type=unet epoches=$epoches seed=$seed train_size=.3
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.4 model_type=unet epoches=$epoches seed=$seed train_size=.4
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.5 model_type=unet epoches=$epoches seed=$seed train_size=.5
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.6 model_type=unet epoches=$epoches seed=$seed train_size=.6
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.7 model_type=unet epoches=$epoches seed=$seed train_size=.7
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.8 model_type=unet epoches=$epoches seed=$seed train_size=.8
python train_reduced.py with train_on=MAPS logdir=results/unet_model_trained_on_MAPS_.9 model_type=unet epoches=$epoches seed=$seed train_size=.9

python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.1 model_type=unet epoches=$epoches seed=$seed train_size=.1
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.2 model_type=unet epoches=$epoches seed=$seed train_size=.2
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.3 model_type=unet epoches=$epoches seed=$seed train_size=.3
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.4 model_type=unet epoches=$epoches seed=$seed train_size=.4
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.5 model_type=unet epoches=$epoches seed=$seed train_size=.5
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.6 model_type=unet epoches=$epoches seed=$seed train_size=.6
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.7 model_type=unet epoches=$epoches seed=$seed train_size=.7
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.8 model_type=unet epoches=$epoches seed=$seed train_size=.8
python train_reduced.py with train_on=GuitarSet logdir=results/unet_model_trained_on_GuitarSet_.9 model_type=unet epoches=$epoches seed=$seed train_size=.9

python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.1 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.1
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.2 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.2
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.3 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.3
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.4 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.4
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.5 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.5
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.6 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.6
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.7 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.7
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.8 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.8
python train_reduced.py with train_on=MAPS logdir=results/transferred_unet_model_trained_on_MAPS_.9 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.9

python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.1 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.1
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.2 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.2
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.3 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.3
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.4 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.4
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.5 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.5
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.6 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.6
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.7 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.7
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.8 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.8
python train_reduced.py with train_on=GuitarSet logdir=results/transferred_unet_model_trained_on_GuitarSet_.9 model_type=unet pretrained_model_path=results/unet_model_trained_on_SynthesizedInstruments/model-$epoches.pt epoches=$epoches seed=$seed train_size=.9
