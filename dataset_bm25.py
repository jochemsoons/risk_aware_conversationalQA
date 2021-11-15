import json
import csv
import re
import glob
import random
from rank_bm25 import BM25Okapi

class ConversationDataset_BM25():
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
        files_in_batch = 0
        data_ids, conversations, conversations_tokenized = [],[], []
        seen_indexes = []

        # append data ids, conversations and tokenized conversations to lists
        for data_file in all_data_list:
            f = open(data_file)
            data = f.readlines()
            data = [d.strip() for d in data]
            data_id = data_file.split('/')[-1]
            data_ids.append(data_id)
            conversations.append(data)

            # use whole conversation to obtain negative instances
            sentences_tokenized = []
            for sentence in data:
                sentences_tokenized.append(sentence.split(" "))
            sentences_tokenized_flat = [item for sublist in sentences_tokenized for item in sublist]
            conversations_tokenized.append(sentences_tokenized_flat)

        # calculate bm25 embeddings for all conversations
        bm25 = BM25Okapi(conversations_tokenized)

        # loop over conversations
        for i in range(len(conversations)):

            # tokenize query and corpus and calculate best matching documents given BM25
            tokenized_query = conversations_tokenized[i]
            doc_scores = bm25.get_scores(tokenized_query)
            best_scores_indexes = (-doc_scores).argsort()

            # create batches of positive and best BM25 negatives
            batch = {'conversations':{}, 'responses_pool':[], 'answers_pool':[]}
            for j in range(batch_size):
                
                # create batch with conversations, responses and answers
                index = best_scores_indexes[j]
                data_id = data_ids[index]
                data = conversations[index]
                batch['conversations'][data_id] = data
                for ut_num in range(len(data)):
                    if ut_num % 2 and ut_num != (len(data) - 1) :
                        batch['responses_pool'].append(data[ut_num])
                batch['answers_pool'].append(data[-1])
            self.batches.append(batch)
