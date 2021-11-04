import json
import csv
import re
import glob
import random
from rank_bm25 import BM25Okapi

class ConversationDataset():
    '''
    The conversation database class. 
    '''
    def __init__(self, path_to_dataset, batch_size, max_size):
        self.batches = []
        self.max_len = 512
        print("Reading data from", path_to_dataset, "batch size", batch_size)
        all_data_list = glob.glob(path_to_dataset + '*')
        all_data_list.sort()
        all_data_list = all_data_list[:max_size] # max size
        random.shuffle(all_data_list) # ADDED (SHUFFLE DATA BEFORE MAKING BATCHES)
        files_in_batch = 0
        data_ids, conversations, conversations_tokenized = [],[], []

        # append data ids, conversations and tokenized conversations to lists
        for data_file in all_data_list:
            f = open(data_file)
            data = f.readlines()
            data = [d.strip() for d in data]
            data_id = data_file.split('/')[-1]
            data_ids.append(data_id)
            conversations.append(data)
            conversations_tokenized.append(data[0].split(" "))
        
        # loop over conversations
        for i in range(len(conversations)):
            print('creating batch 'i, ,'/',len(conversations))

            # tokenize query and corpus and calculate best matching documents given BM25
            batch_full = False
            tokenized_query = conversations_tokenized[i]
            tokenized_corpus = conversations_tokenized
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(tokenized_query)
            best_scores_indexes = (-doc_scores).argsort()

            # create batches of positive and best BM25 negatives (under restriction that negative can only occur 1 time in all batches)
            while not batch_full:
                for index in best_scores_indexes:

                    if files_in_batch == 0:
                        self.batches.append({'conversations':{}, 'responses_pool':[], 'answers_pool':[]})
                    
                    # create batch with conversations, responses and answers
                    data_id = data_ids[index]
                    data = conversations[index]
                    self.batches[-1]['conversations'][data_id] = data
                    for ut_num in range(len(data)):
                        if ut_num % 2 and ut_num != (len(data) - 1) :
                            self.batches[-1]['responses_pool'].append(data[ut_num])
                    self.batches[-1]['answers_pool'].append(data[-1])

                    # if batch has 10 conversations then make a new batch
                    files_in_batch += 1
                    if files_in_batch == batch_size:
                        batch_full = True
                        files_in_batch = 0
