from math import log
from read_input import load_data
import sys
import json
import ast

if __name__ == '__main__':
    argument = sys.argv
    test_directory =  argument[1]

    #test_directory = "C:\\CSCI544-NaturalLanguageProcessing\\Homework\\HW2\\test\\it_isdt_dev_raw.txt"
    # load the test data
    test_data = load_data(test_directory)

    # load model
    with open('hmmmodel.txt') as json_file:
        data = json.load(json_file)
    
    # load word set
    with open('word_set.txt','r',encoding='utf-8') as f:
        word_set = ast.literal_eval(f.read())

    # load the emission and transition dictionaries
    emission = data[0]
    transition = data[1]

    transition_probability = {}
    # find transition probabilty of one tag to next tag
    for prev_tag in transition:
        transition_probability[prev_tag] = {}
        total_count = sum(transition[prev_tag].values())
        for tag in transition[prev_tag]:
            transition_probability[prev_tag][tag] = transition[prev_tag][tag]/total_count

    # viterbi decoding
    for s,sentence in enumerate(test_data):
        # stores the probability of a word belonging to each tag
        path = []
        # stores the most likely sequence of tags
        path_tags = []

        for index,word in enumerate(sentence):
            path.append({})
            path_tags.append({})

            # first word of sentence -- no prior path probability, only emission and transition probabilities
            if index==0:
                for tag in emission:
                    if word.isdigit():
                        emission_prob = log(emission[tag].get('<num_tag>',1e-6)/sum(emission[tag].values()))

                    else:
                        emission_prob = log(emission[tag].get(word,1e-6)/sum(emission[tag].values()))
                    transition_prob = log(transition_probability['start_sentence'].get(tag))
                    path[index][tag] = emission_prob + transition_prob
                    path_tags[index][tag] = 'start_sentence'
            else:                
                # traverse all the possible tags
                for tag in emission:
                    path[index][tag] = -1000000

                    # if word is a number calculated emission prob of tags with num_tag
                    if word.isdigit():
                        emission_prob = emission[tag].get('<num_tag>',0)/sum(emission[tag].values())
                    
                    elif word in word_set:
                        emission_prob = emission[tag].get(word,0)/sum(emission[tag].values())
                    
                    # for unseen word calculate emission probabilty of tags having low freq words
                    else:
                        emission_prob = emission[tag].get('<unk_tag>',0)/sum(emission[tag].values())

                    # if word is never seen as a particular tag -- skip the tag
                    if emission_prob == 0:
                        continue
                    emission_prob = log(emission_prob)

                    # find highest likely previous tag by checking all previous tags that have a non zero path
                    for prev_tag in path_tags[index-1]:
                        transition_prob = log(transition_probability[prev_tag].get(tag))
                        joint_prob = emission_prob + transition_prob + path[index-1][prev_tag]
                        if path[index][tag] < joint_prob:
                            path[index][tag] = joint_prob
                            path_tags[index][tag] = prev_tag

        # multiply the available paths with end sentence probability 
        for tag in path_tags[index]:
            end_prob = log(transition_probability[tag].get('end_sentence'))
            path[index][tag] += end_prob

        # get the most likely tag sequence ending
        high_prob_tag = max(path[-1], key=path[-1].get)

        # assign each word the most likely tag by following the most likely tag path
        sentence_tags = []
        for i in range(len(path)-1,-1,-1):
            word_tag = sentence[i] + '/' + high_prob_tag
            sentence_tags.insert(0,word_tag)
            high_prob_tag = path_tags[i][high_prob_tag]
        test_data[s] = sentence_tags

    with open('hmmoutput.txt', 'w',encoding='utf-8') as f:
        for sentence in test_data:
            for i,word in enumerate(sentence):
                if i==len(sentence)-1:
                    f.write('%s' % word)
                    continue    
                f.write('%s ' % word)
            f.write('\n')


