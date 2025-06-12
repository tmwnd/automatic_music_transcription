import copy
import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fail_observer import FailObserver
from model import *
from model.constants import *
from model.dataset import (OriginalMAPS, SynthesizedInstruments,
                           SynthesizedTrumpet, CustomBatchDataset)
from model.evaluate_fn import evaluate_wo_velocity
from snapshot import Snapshot

ex = Experiment('train_transcriber')


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# parameters for the network
ds_ksize, ds_stride = (2, 2), (2, 2)
mode = 'imagewise'
sparsity = 1

@ex.config
def config():
    # Choosing GPU to use
    GPU = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    device = 'cuda'
    dataset_root_dir = "."
    spec = 'CQT'
    resume_iteration = None
    train_on = 'GuitarSet'
    pretrained_model_path = None
    freeze_all_layers = False
    unfreeze_linear = False
    unfreeze_lstm = False
    debug_mode = False # caution - it does terrible things

    batch_size = 32
    sequence_length = 327680
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(
            f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    epoches = 2000
    learning_rate = 0.01
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98
    model_type = "unet"

    leave_one_out = None
    seed = 33 # set seed for whole experiment
    seed_everything(33)
    clip_gradient_norm = 3

    validation_length = sequence_length
    refresh = False
    custom_batch = False
    start_debug_epoch = 1

    train_size = 1

    destination_dir = "runs"
    logdir = f'{destination_dir}/TRAIN_TRANSCRIPTION_{model_type}_ON_{train_on}_{spec}_{mode}_' + \
        datetime.now().strftime('%y%m%d-%H%M%S')
    fail_observer = FailObserver(logdir=logdir, snapshot_capacity=10)
    ex.observers.append(FileStorageObserver.create(logdir))
    ex.observers.append(fail_observer)


def detect_epoch(filename):
    only_model_name = filename.split("/")[-1]
    return only_model_name[6:-3]

def get_gradients(model):
	grads = []
	for params in model.parameters():
		grads.append(torch.max(torch.abs(params.grad)).item())
	return grads

def create_transcription_datasets(dataset_type):
    if dataset_type == "MAESTRO":
        return [(MAESTRO, ["train"]), (MAESTRO, ["validation"]), (MAESTRO, ["test"])]
    elif dataset_type == "MusicNet":
        return [(MusicNet, ['train']), (MusicNet, ['test']), (MAPS, ['ENSTDkAm', 'ENSTDkCl'])]
    elif dataset_type == "GuitarSet":
        return [(GuitarSet, ['train']), (GuitarSet, ['val']), (GuitarSet, ['test'])]
    elif dataset_type == "SynthesizedTrumpet":
        return [(SynthesizedTrumpet, ['train']), (SynthesizedTrumpet, ['val']), (SynthesizedTrumpet, ['test'])]
    elif dataset_type == "SynthesizedInstruments":
        return [(SynthesizedInstruments, ['train']), (SynthesizedInstruments, ['val']), (SynthesizedInstruments, ['test'])]
    elif dataset_type == "MAPS":
        return [(MAPS, ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']),
                (MAPS, ['ENSTDkAm', 'ENSTDkCl']),
                (MAPS, ['ENSTDkAm', 'ENSTDkCl'])]
    elif dataset_type == "OriginalMAPS":
        return [(OriginalMAPS, ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']),
                (OriginalMAPS, ['ENSTDkAm', 'ENSTDkCl']),
                (OriginalMAPS, ['ENSTDkAm', 'ENSTDkCl'])]


def create_model(model_type):
    if model_type == "resnet":
        print("Using resnet transcription model")
        return ResnetTranscriptionModel
    else:  # fallback for unet
        print("Using unet transcription model")
        return UnetTranscriptionModel

def make_dir_if_does_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_4D_array_to_file(filename, array_to_be_saved):
    for i, element in enumerate(array_to_be_saved):
        if(array_to_be_saved.shape[1] == 1):
            np.savetxt(filename+f"_{i}.txt", element[0])
        else:
            for j, subelement in enumerate(element):
                np.savetxt(filename+f"_{i}_{j}.txt", subelement)
def serialize_batch(batch, batch_idx, batch_saving_destination):
        batch_audio = batch['audio'].clone().detach().cpu().numpy()
        batch_frame = batch['frame'].clone().detach().cpu().numpy()
        batch_path = np.asarray(batch['path'])
        np.save(os.path.join(batch_saving_destination, f"batch_audio.npy"), batch_audio)
        np.save(os.path.join(batch_saving_destination, f"batch_frame.npy"), batch_frame)
        np.save(os.path.join(batch_saving_destination, f"batch_path.npy"), batch_path)

def print_layers_names(model):
    id = 0
    for param, _ in model.named_parameters():
        print(f"{id}: {param}")
        id = id+1

def save_batch(batch, logdir, ep, batch_idx):
    batch_saving_destination = os.path.join(logdir, f"batch/{ep}/{batch_idx}/")
    make_dir_if_does_not_exist(batch_saving_destination)
    serialize_batch(batch, batch_idx, batch_saving_destination)

def save_gradients(model, logdir, ep, batch_idx):
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            if idx in [66,67,70,71,76,77,80,81,84,86,87,88,96,97,98,99]:
                destination_dir = os.path.join(logdir, f"gradients/{ep}/{batch_idx}/layer_{idx}")
                make_dir_if_does_not_exist(destination_dir)
                detached_grad = param.grad.data.clone().detach().cpu().numpy()
                basic_filename = os.path.join(destination_dir, "weight")
                if(len(detached_grad.shape) == 4):
                    save_4D_array_to_file(basic_filename, detached_grad)
                else:
                    np.savetxt(basic_filename+".txt", detached_grad)

@ex.automain
def train(spec, resume_iteration, train_on, pretrained_model_path, freeze_all_layers, unfreeze_linear, unfreeze_lstm,
          batch_size, sequence_length, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate,
          leave_one_out, clip_gradient_norm, validation_length, refresh, device, epoches, logdir,
          dataset_root_dir, model_type, fail_observer, debug_mode, custom_batch, start_debug_epoch, train_size):
    print_config(ex.current_run)
    dataset_data = create_transcription_datasets(dataset_type=train_on)
    TrainDataset = dataset_data[0][0]
    train_dataset_groups = dataset_data[0][1]
    ValidationDataset = dataset_data[1][0]
    val_dataset_groups = dataset_data[1][1]
    TestDataset = dataset_data[2][0]
    test_dataset_groups = dataset_data[2][1]
    if not custom_batch:
        train_dataset = TrainDataset(dataset_root_dir=dataset_root_dir, groups=train_dataset_groups,
                                 sequence_length=sequence_length, device=device, refresh=refresh, train_size=train_size)
    # validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    validation_dataset = ValidationDataset(dataset_root_dir=dataset_root_dir, groups=val_dataset_groups,
                                           sequence_length=sequence_length, device=device, refresh=refresh, train_size=train_size)
    test_dataset = TestDataset(dataset_root_dir=dataset_root_dir, groups=test_dataset_groups,
                               sequence_length=sequence_length, device=device, refresh=refresh, train_size=train_size)
    if custom_batch:
        loader = CustomBatchDataset(dataset_root_dir="results_monday/TRAIN_TRANSCRIPTION_unet_ON_MAPS_CQT_imagewise_230401-003349/batch", groups=train_dataset_groups,
                                 sequence_length=sequence_length, device=device, refresh=refresh, train_size=train_size)
    else:
        loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    valloader = DataLoader(validation_dataset, 4,
                           shuffle=False, drop_last=True)
    # Getting one fixed batch for visualization
    batch_visualize = next(iter(valloader))

    if resume_iteration is None:
        ModelClass = create_model(model_type)
        model = ModelClass(ds_ksize=ds_ksize, ds_stride=ds_stride, mode=mode,
                           spec=spec, norm=sparsity, device=device, logdir=logdir, debug_mode=debug_mode)
        model.to(device)
        if pretrained_model_path != None:
            pretrained_model_path = pretrained_model_path
            print("Copying from ", pretrained_model_path)
            pretrained_model = torch.load(pretrained_model_path)
            model.load_my_state_dict(pretrained_model)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            detected_epoch = detect_epoch(pretrained_model_path)
            try:
                optimizer.load_state_dict(torch.load(
                    os.path.join(os.path.dirname(pretrained_model_path), f'last-optimizer-state-{detected_epoch}.pt')))
            except:
                print("Cannot load optimizer! Probably model changed - new optimizer will be created")
                optimizer = optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            resume_iteration = 0
        if freeze_all_layers:
            model.freeze_all_layers()
        if unfreeze_linear or unfreeze_lstm:
            model.unfreeze_selected_layers(
                linear=unfreeze_linear, lstm=unfreeze_lstm)

    else:  # Loading checkpoints and continue training
        trained_dir = 'trained_MAPS'  # Assume that the checkpoint is in this folder
        model_path = os.path.join(trained_dir, f'{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(
            os.path.join(trained_dir, 'last-optimizer-state.pt')))

    summary(model)
    if debug_mode:
        print_layers_names(model)
        torch.autograd.set_detect_anomaly(True)
    scheduler = StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    # loop = tqdm(range(resume_iteration + 1, iterations + 1))
    total_batch = len(loader.dataset)

    for ep in range(1, epoches+1):
        model.train()
        total_loss = 0
        batch_idx = 0
        if not model.internal_debug_mode and debug_mode and ep == start_debug_epoch:
            print("Started to register batches when problem occurs!")
            print(f"epoch: {ep}, first batch: {batch_idx}")
            model.internal_debug_mode = True
        # print(f'ep = {ep}, lr = {scheduler.get_lr()}')
        for batch in loader:
            optimizer.zero_grad()
            predictions, losses, _ = model.run_on_batch(batch, batch_description=str(ep), batch_identifier=batch_idx)
            loss = sum(losses.values())
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            if model.internal_debug_mode:
                save_batch(batch, logdir, ep, batch_idx)
                save_gradients(model, logdir, ep, batch_idx)
            optimizer.step()
            scheduler.step()
            batch_idx += 1
            print(f'Train Epoch: {ep} [{batch_idx*batch_size}/{total_batch}'
                  f'({100. * batch_idx*batch_size / total_batch:.0f}%)]'
                  f'\tLoss: {loss.item():.6f}')

        epoch_loss = total_loss/len(loader)
        print(f'Train Epoch: {ep}\tLoss: {epoch_loss:.6f}')
        fail_observer.snapshot.add_to_snapshot(ep,
                                               epoch_loss,
                                               f"Epoch: {ep} Loss: {epoch_loss}",
                                               copy.deepcopy(model.state_dict()),
                                               copy.deepcopy(optimizer.state_dict()),
                                               copy.deepcopy(scheduler.state_dict()))
        if fail_observer.snapshot.snapshot_triggered:
            print("The end of training! Snapshot is taken!")
            break
        # Logging results to tensorboard
        if ep == 1:
            #             os.makedirs(logdir, exist_ok=True) # creating the log dir
            writer = SummaryWriter(logdir)  # create tensorboard logger

        if (ep) % 10 == 0 and ep > 1:
            model.eval()
            with torch.no_grad():
                for key, values in evaluate_wo_velocity(validation_dataset, model).items():
                    if key.startswith('metric/'):
                        _, category, name = key.split('/')
                        print(
                            f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                        if ('precision' in name or 'recall' in name or 'f1' in name or 'levensthein' in name) and 'chroma' not in name:
                            writer.add_scalar(
                                key, np.mean(values), global_step=ep)

        if (ep) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(
                logdir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(
                logdir, f'last-optimizer-state-{ep}.pt'))
        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=ep)

        # Load one batch from validation_dataset

        predictions, losses, mel = model.run_on_batch(batch_visualize, "evaluation")
        if ep == 1:  # Showing the original transcription and spectrograms
            fig, axs = plt.subplots(2, 2, figsize=(24, 8))
            axs = axs.flat
            for idx, i in enumerate(mel.cpu().detach().numpy()):
                axs[idx].imshow(i.transpose(), cmap='jet', origin='lower')
                axs[idx].axis('off')
            fig.tight_layout()

            writer.add_figure('images/Original', fig, ep)

            fig, axs = plt.subplots(2, 2, figsize=(24, 4))
            axs = axs.flat
            for idx, i in enumerate(batch_visualize['frame'].cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Label', fig, ep)

        if ep < 11 or (ep % 50 == 0):
            fig, axs = plt.subplots(2, 2, figsize=(24, 4))
            axs = axs.flat
            for idx, i in enumerate(predictions['frame'].detach().cpu().numpy()):
                axs[idx].imshow(i.transpose(), origin='lower', vmax=1, vmin=0)
                axs[idx].axis('off')
            fig.tight_layout()
            writer.add_figure('images/Transcription', fig, ep)

            fig, axs = plt.subplots(2, 2, figsize=(24, 8))
            if model_type != "resnet":
                axs = axs.flat
                for idx, i in enumerate(predictions['feat1'].detach().cpu().numpy()):
                    axs[idx].imshow(i[0].transpose(), cmap='jet',
                                    origin='lower', vmax=1, vmin=0)
                    axs[idx].axis('off')
                fig.tight_layout()
                writer.add_figure('images/feat1', fig, ep)

    # Evaluating model performance on the full MAPS songs in the test split
    print('Training finished, now evaluating ')
    with torch.no_grad():
        model = model.eval()
        metrics = evaluate_wo_velocity(tqdm(test_dataset), model,
                                       save_path=os.path.join(logdir, './MIDI_results'))

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(
                f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')

    export_path = os.path.join(logdir, 'result_dict')
    pickle.dump(metrics, open(export_path, 'wb'))
