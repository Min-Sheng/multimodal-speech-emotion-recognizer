import os
import time
import datetime

import torch
import numpy as np
import torch.nn as nn
from torch import optim
from data_loader.single_text_loader import SingleTextLoader
from model.single_text_model import SingleTextModel
from config.single_text_config import *    
from tensorboardX import SummaryWriter

def run_eval(model, batch_gen, data, device, step, valid_type):
    
    sum_batch_ce = 0
    list_batch_correct = []
    
    list_pred = []
    list_label = []

    max_loop  = int(len(data) / BATCH_SIZE)
    remaining = int(len(data) % BATCH_SIZE)

    # evaluate data ( N of chunk (batch_size) + remaining( +1) )
    for test_itr in range( max_loop + 1 ):
        model.eval()
        with torch.no_grad():
            input_seq, input_lengths, labels = batch_gen.get_batch(
                                            data=data,
                                            batch_size=BATCH_SIZE,
                                            encoder_size=ENCODER_SIZE,
                                            is_test=True,
                                            start_index= (test_itr* BATCH_SIZE)
                                            )
            input_seq = input_seq.to(device)
            input_lengths = input_lengths.to(device)
            labels = labels.to(device)
            
            _, labels = labels.max(dim=1)
            
            predictions = model(input_seq, input_lengths)
            predictions = predictions.to(device)
            loss = criterion(predictions, labels)
            
            # remaining data case (last iteration)
            if test_itr == (max_loop):
                predictions = predictions[:remaining]
                loss = loss[:remaining]
                labels = labels[:remaining]
                
            # evaluate on cpu
            loss = torch.sum(loss)
            sum_batch_ce += loss.item()
            predictions = np.array(predictions.cpu())
            labels = np.array(labels.cpu())
            
            # batch accuracy
            list_pred.extend(np.argmax(predictions, axis=1))
            #list_label.extend(np.argmax(labels, axis=1))
            list_label.extend(labels)
            
    list_batch_correct = [1 for x, y in zip(list_pred, list_label) if x==y]
    accr = np.sum (list_batch_correct) / float(len(data))
    ce = sum_batch_ce / float(len(data))
    if valid_type=='valid':
        writer.add_scalar('loss/valid_loss', ce, step)
        writer.add_scalar('accuracy/valid_accuracy', accr, step)
    elif valid_type=='test':
        writer.add_scalar('loss/test_loss', ce, step)
        writer.add_scalar('accuracy/test_accuracy', accr, step)
    return ce, accr
        
if __name__ == '__main__':
    
    device = 'cuda:{}'.format(GPU) if \
             torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter('runs/'+MODEL_NAME)
    batch_gen = SingleTextLoader()
    
    model = SingleTextModel(dic_size = batch_gen.dic_size, 
                         use_glove = USE_GLOVE, 
                         num_layers = N_LAYER,
                         hidden_dim = HIDDEN_DIM, output_dim = N_CATEGORY, 
                         embedding_dim = DIM_WORD_EMBEDDING, dr = DROPOUT_RATE, 
                         bidirectional = BIDIRECTIONAL, embedding_train=EMBEDDING_TRAIN)
    
    model = model.to(device)
    #criterion = nn.BCEWithLogitsLoss(reduction='none')
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-05, verbose=True)
    
    early_stop_count = MAX_EARLY_STOP_COUNT
    valid_freq = int(len(batch_gen.train_set) * EPOCH_PER_VALID_FREQ / float(BATCH_SIZE)) + 1
    print("[Info] Valid Freq = " + str(valid_freq))

    initial_time = time.time()
    min_ce = 1000000
    train_ce = 0
    
    for step in range(N_STEPS):
        model.train()
        input_seq, input_lengths, labels = batch_gen.get_batch(
                                        data=batch_gen.train_set,
                                        batch_size=BATCH_SIZE,
                                        encoder_size=ENCODER_SIZE,                                        
                                        is_test=False
                                        )
        input_seq = input_seq.to(device)
        input_lengths = input_lengths.to(device)
        labels = labels.to(device)
        
        _, labels = labels.max(dim=1)
        
        model.zero_grad()
        optimizer.zero_grad()

        predictions = model(input_seq, input_lengths)
        predictions = predictions.to(device)
        loss = criterion(predictions, labels)
        loss = torch.mean(loss)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        writer.add_scalar('loss/train_loss', loss.item(), step)
        train_ce += loss.item()
        
        if (step + 1) % valid_freq == 0:
            
            dev_ce, dev_accr = run_eval(model=model, batch_gen=batch_gen, data=batch_gen.dev_set, device=device, step = step, valid_type = 'valid')
            scheduler.step(dev_ce)
            end_time = time.time()
            
            if step > CAL_ACCURACY_FROM:
                
                test_ce, test_accr = run_eval(model=model, batch_gen=batch_gen, data=batch_gen.test_set, step = step, device=device, valid_type = 'test')
            
                if MAX_EARLY_STOP_COUNT != -1:
                    if dev_ce < min_ce:
                        min_ce = dev_ce
                        early_stop_count = MAX_EARLY_STOP_COUNT

                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print("early stopped")
                            torch.save(model, 'save/' + MODEL_NAME + '_model.pkl')
                            break

                        early_stop_count = early_stop_count -1
                else:
                    print('{:.3f}'.format(int(end_time - initial_time)/60) + " mins" + \
                        " step/seen/itr: " + str(step) + "/ " + \
                                             str(step * BATCH_SIZE) + "/" + \
                                             str(round(step * BATCH_SIZE / float(len(batch_gen.train_set)), 2)) + \
                        "\tdev_acc: " + '{:.6f}'.format(dev_accr)  + "  test_acc: " + '{:.6f}'.format(test_accr) + "  dev_loss: " + '{:.6f}'.format(dev_ce) + "  train_loss: " + '{:.6f}'.format(train_ce/(valid_freq-1)))
                    train_ce = 0
    print("end training.")
    torch.save(model, 'save/' + MODEL_NAME + '_model.pkl')