import sys
from read_input import load_data
from collections import Counter, defaultdict
import json

def create_model(train_data):
    # find low frequency words in the train data set
    lf_words = find_lf_words(train_data, 1)
    # different words in the train data set
    word_set = set()

    # emission and transition dictionaries
    emission = defaultdict(lambda: defaultdict(int))
    transition = defaultdict(lambda: defaultdict(int))

    for sentence in train_data:
        # first previous tag is an additional tag just to show start of sentence
        prev_tag = 'start_sentence'

        for word_tag in sentence:
            # split the word-tag by splitting just once using the rightmost /
            word,tag = word_tag.rsplit('/',1)

            # add word in set of words
            word_set.add(word)

            # if word is a number then increment num_tag by 1 for the tag associated with the number 
            if word.isdigit():
                emission[tag]['<num_tag>'] += 1

            # increase count of word for the tag by 1
            emission[tag][word] +=  1

            # if word is a low freq word then increment unk_tag for associated tag
            if word in lf_words:
                emission[tag]['<unk_tag>'] +=1
            
            # increase count of tag for the previos tag by 1
            transition[prev_tag][tag] +=  1

            # update previous tag to current tag
            prev_tag = tag            
            
        # get count of which tags are last in sentence
        transition[prev_tag]['end_sentence'] +=  1

    transition['end_sentence'] = {}

    # +1 smoothing for transitions not encountered from the train data
    for prev_tag in transition:
        for tag in transition:
            transition[prev_tag][tag] = transition[prev_tag].get(tag,0) + 1

    return emission, transition, word_set

def find_lf_words(train_data, min_threshold=1):
    low_freq_words = set()
    word_count = Counter()
    # find count of each word in the train data set
    for sentence in train_data:
        for word_tag in sentence:
            word,tag = word_tag.rsplit('/',1)

            word_count.update([word])
    
    # if count of word less than threshold the it's a low freq word
    for word, count in word_count.most_common()[::-1]:
        if count>min_threshold:
            break
        low_freq_words.add(word)
    
    return low_freq_words


def write_model(emission, transition, word_set):
    matrices = []
    matrices.append(emission)
    matrices.append(transition)
    with open('hmmmodel.txt', 'w') as fp:
        json.dump(matrices, fp)
    with open('word_set.txt','w',encoding='utf-8') as f:
        f.write(str(word_set))

if __name__ == '__main__':
    argument = sys.argv
    train_directory =  argument[1]
    # train_directory = "C:\\CSCI544-NaturalLanguageProcessing\\Homework\\HW2\\train\\it_isdt_train_tagged.txt"

    # load the training data
    train_data = load_data(train_directory)

    # create emission and transition matrices from the train data
    emission, transition, word_set = create_model(train_data)

    # output the emission and transition matrices in a file
    write_model(emission,transition,word_set)