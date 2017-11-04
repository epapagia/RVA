# -*- coding: utf-8 -*-
from os import listdir
import os
import sys
import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import operator
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from nltk import ngrams
from nltk.corpus import wordnet
# this example uses TopicRank
from pke import TopicRank 
import matplotlib.pyplot as plt
import numpy as np
import argparse
from gensim.models import KeyedVectors

import sys
import codecs
from sklearn.manifold import TSNE

from scipy.spatial.distance import canberra
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import sqeuclidean
from scipy.spatial.distance import correlation
import subprocess
from subprocess import call


_STOP_WORDS = [
'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 
'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although',
'always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another',
'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',
'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been', 
'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 
'between', 'beyond', 'both', 'bottom','but', 'by', 'can', 
'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 
'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 
'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 
'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 
'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 
'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 
'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 
'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'iff', 'in', 'inc', 
'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 
'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 
'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 
'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 
'nevertheless', 'next', 'nine', 'no', 'non', 'nobody', 'none', 'noone', 'nor', 'not', 
'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only',
'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out',
'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same',
'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 
'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 
'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 
'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 
'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 
'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 
'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 
'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 
'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 
'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 
'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
'yourselves', 'the', 'zj', 'zi', 'yj', 'yi', 'xi', 'xj', 'xixj', 'xjxi', 'yiyj', 
'yjyi', 'zizj', 'zjzi']

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(ps.stem(filter(lambda y: y in string.printable, item)))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, ps)
    return stems

stop = set(stopwords.words('english'))

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS', 'FW']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wordnet.ADJ
    elif is_noun(tag):
        return wordnet.NOUN
    elif is_adverb(tag):
        return wordnet.ADV
    elif is_verb(tag):
        return wordnet.VERB
    return wordnet.NOUN

_WORD_MIN_LENGTH = 3
_WORD_MAX_LENGTH = 35
_NUM_ITERATIONS = 50#3000
_DIM_VECTOR = 50
_FREQUENCY = 1
_PURE_VECTOR = False

def generate(vocab_file, vectors_file):

    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


data_original_path = '/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin2009/all_docs_abstacts_refined/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin2009/all_docs_abstacts_refined/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/SemEval2010-Maui/original/SemEval2010/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin2009/all_docs_abstacts_refined/'

abstracts_path = '/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin_Abstracts/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin2009/experiment_abstracts/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin2009/experiment_abstracts/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/SemEval2010-Maui/experiment_abstracts/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Semeval2010_Abstracts/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin_Abstracts/'

print 'Read data files...'
data_original_files = [f for f in listdir(data_original_path)]
abstract_files = [f for f in listdir(abstracts_path)]

ps = PorterStemmer()
vdoc_keys = {}   
prec_test = []
rec_test = []
fm_test = []
prec_test2 = []
rec_test2 = []
fm_test2 = [] 
prec_test_strict = []
rec_test_strict = []
fm_test_strict = [] 
counter_of_files = 0
labels = []
for fv, filev in enumerate(data_original_files):
#    print filev
#    filev = '305093.txt'
    W, vocab, ivocab = generate("vocab.txt"+filev.replace('.key','.txt')+str(_DIM_VECTOR)+str(_NUM_ITERATIONS), "vectors"+filev.replace('.key','.txt')+str(_DIM_VECTOR)+str(_NUM_ITERATIONS)+'.txt')
    keyphrases = []
    if filev.endswith(".txt") and filev.replace('.txt','.abstr') in abstract_files:
        labels.append(filev.replace('.txt',''))

        print 'Find keyphrases for file', filev, '...'
        with open(data_original_path+filev.replace('.txt','.key'), 'r') as fkey:   
            for line in fkey:
                line_words = line.replace('\n', '').replace('\r', '').lower().split(" ")
                keyw = ''
                for lw in line_words:
                    lw = filter(lambda y: y in string.printable, lw).lower()
                    keyw += ps.stem(lw.decode('latin-1').encode("utf-8").decode('utf-8'))+' ' 
                if str(keyw).translate(None, string.punctuation).strip().split(' ') not in keyphrases:
                    keyphrases.append(str(keyw).translate(None, string.punctuation).strip().split(' '))
        vdoc_keys[filev.replace('.txt','.key')] = keyphrases    
        print vdoc_keys[filev.replace('.txt','.key')]
        
        counter_of_files += 1
        print counter_of_files, 'Tokens for file', filev.replace('.key','.txt'), '...'
        dict_word_pos = {}
        with open(abstracts_path+filev.replace('.txt','.abstr'), 'r') as myfile:
            text = myfile.read()
#            sep_words = [next(myfile) for x in xrange(20)]
#            text = ' '.join(sep_words)
            lowers = filter(lambda y: y in string.printable, text).lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            temp = no_punctuation.split(' ')
            tempn = nltk.word_tokenize(no_punctuation)
            tagged = nltk.pos_tag(tempn)
            tokens_stemmed = []
            for x in temp:
                tokens_stemmed.append(ps.stem(filter(lambda y: y in string.printable, x)))
            for x in tagged:
                #0 position is a, 1 position is r, 2 position is v
                if ps.stem(x[0]) not in dict_word_pos.keys():    
                    dict_word_pos[ps.stem(x[0])] = penn_to_wn(x[1])
                else:
                    tmpp = dict_word_pos[ps.stem(x[0])]+penn_to_wn(x[1])
                    dict_word_pos[ps.stem(x[0])] = tmpp
            #print dict_word_pos    
        doc_unigrams = []
        doc_unigrams_stemmed = []
        tokens = []
        tokens_stemmed_real = []
        bbigrams = []
        ttrigrams = []
        no_punctuation_stemmed = ''
        no_punctuation_unstemmed = ''
        #print 'Find all candidate unigrams, bigrams and trigrams for',filev.replace('.key','.txt')     
        for x in tempn:
            no_punctuation_stemmed += ps.stem(filter(lambda y: y in string.printable, x))+' '
            no_punctuation_unstemmed += filter(lambda y: y in string.printable, x)+' '
            tokens.append(filter(lambda y: y in string.printable, x))
        for token in tokens:
            token = token.strip().lower()
            token = token.strip(string.digits)
            if len(token) >= _WORD_MIN_LENGTH and len(token) <= _WORD_MAX_LENGTH and '!' not in token and '@' not in token and '#' not in token and '$' not in token and '*' not in token and '=' not in token and '+' not in token and '\\x' not in token and '.' not in token and ',' not in token and '?' not in token and '>' not in token and '<' not in token and '&' not in token and not token.isdigit() and token not in _STOP_WORDS and token not in stop and '(' not in token and ')' not in token and '[' not in token and ']' not in token and '{' not in token and '}' not in token and '|' not in token and token not in doc_unigrams:  #and tokens_stemmed.count(ps.stem(token))>=_FREQUENCY                                                      
                doc_unigrams.append(token)
                doc_unigrams_stemmed.append(ps.stem(token))
        print 'number of unigrams', len(doc_unigrams), doc_unigrams
        n = 2
        bigrams = ngrams(tokens, n)
        for bi in bigrams:
            token1 = filter(lambda x: x in string.printable, bi[0])
            token1 = token1.lower()
            token2 = filter(lambda x: x in string.printable, bi[1])
            token2 = token2.lower()
            if token1 in doc_unigrams and token2 in doc_unigrams and not (len(token1)<=3 and len(token2)<=3):
                big = filter(lambda y: y in string.printable, token1.lower())+' '+filter(lambda y: y in string.printable, token2.lower())
                bitu = (filter(lambda y: y in string.printable, token1.lower()), filter(lambda y: y in string.printable, token2.lower()))
                if no_punctuation_unstemmed.count(big.strip())>=(_FREQUENCY):
                    bbigrams.append(bitu)
        print 'number of bigrams', len(bbigrams)
        n = 3
        trigrams = ngrams(tokens, n)
        for tri in trigrams:
            token1 = filter(lambda x: x in string.printable, tri[0])
            token1 = token1.lower()
            token2 = filter(lambda x: x in string.printable, tri[1])
            token2 = token2.lower()
            token3 = filter(lambda x: x in string.printable, tri[2])
            token3 = token3.lower()
            if token1 in doc_unigrams and token2 in doc_unigrams and token3 in doc_unigrams and not (len(token1)<=3 and len(token2)<=3 and len(token3)<=3):
                big = filter(lambda y: y in string.printable, token1.lower())+' '+filter(lambda y: y in string.printable, token2.lower())+' '+filter(lambda y: y in string.printable, token3.lower())
                tritu = (filter(lambda y: y in string.printable, token1.lower()), filter(lambda y: y in string.printable, token2.lower()), filter(lambda y: y in string.printable, token3.lower()))
                if no_punctuation_unstemmed.count(big.strip())>=(_FREQUENCY):
                    ttrigrams.append(tritu)
        print 'number of trigrams', len(ttrigrams), ttrigrams 
       
        print 'number of unique tokens', len(set(temp)), set(temp)

        _NUM_KEYWORDS = int(len(set(temp))/3)
        count_words = 0
        final_doc_vector = np.zeros((_DIM_VECTOR))
        word_vector = np.zeros((_DIM_VECTOR))
        vectors = []
        words = []
        for word in tempn:
            #print word
            if word in vocab and word in doc_unigrams:# and word not in words:
                word_vector = W[vocab[word], :]
                final_doc_vector += word_vector
                count_words += 1
            
                words.append(word)
                
                vectors.append(word_vector)
                #word_vector = model[word.replace('\n', ' ').strip()]
                #print word_vector
                
        final_doc_vector /= count_words 
        words.append('MEAN_VECTOR')
        vectors.append(final_doc_vector)
        #print final_doc_vector
        
#                vecs = np.array(vectors)
#                
#                
#                fig_size = [0,0]
#                fig_size[0] = 30
#                fig_size[1] = 17    
#                plt.rcParams["figure.figsize"] = fig_size 
#                plt.rcParams.update({'font.size': 12})
#                
#                tsne = TSNE(n_components=2, random_state=0)
#                np.set_printoptions(suppress=True)
#                Y = tsne.fit_transform(vecs[:len(vectors),:])
#             
#                plt.scatter(Y[:, 0], Y[:, 1])
#                flattened = [val for sublist in vdoc_keys[filev] for val in sublist]
#                for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
#                    if ps.stem(label).lower() in flattened:    
#                        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', color = 'red')
#                    elif label=='MEAN_VECTOR':
#                        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', color = 'green')
#                    else:
#                        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#                plt.show()
        
        print 'Calculation of cosine similarity between mean_vec and every word'
        dict_cand_sim = {}
        candidates = []
        similarities = []
        canber = []
        euclid = []
        seuclid = []
        corel = []
        for word in tempn:
            #print word
            if word in vocab and word in doc_unigrams:
                
                word_vector = W[vocab[word], :]  
                #word_vector = model[word.replace('\n', ' ').strip()] 
                #print word_vector 
                #print (np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector)), 'np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector)'                   
                if np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector)>0.0:
                    candidates.append(word)
                    similarities.append(np.dot(final_doc_vector, word_vector)/(np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector)))
                    dict_cand_sim[word] = np.dot(final_doc_vector, word_vector)/(np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector))
                else:
                    candidates.append(word)
                    similarities.append(0.0)
                    dict_cand_sim[word] = np.dot(final_doc_vector, word_vector)/(np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector))
                    
                canber.append(canberra(final_doc_vector, word_vector))
                euclid.append(euclidean(final_doc_vector, word_vector))
                seuclid.append(sqeuclidean(final_doc_vector, word_vector))
                corel.append(correlation(final_doc_vector, word_vector))
        
        kphs = vdoc_keys[filev.replace('.txt','.key')]
        flattened_kphs = [val for sublist in kphs for val in sublist]
        
        print 'cosine similarity'
        keyphr_sim = []
        for f in flattened_kphs:
            if f in candidates:
                #print candidates[candidates.index(f)], similarities[candidates.index(f)]
                keyphr_sim.append(similarities[candidates.index(f)])
            
        not_keyphr_sim = []
        for f in candidates:
            if f not in flattened_kphs:
                #print f, similarities[candidates.index(f)]
                not_keyphr_sim.append(similarities[candidates.index(f)])
        
        
        data_to_plot = [keyphr_sim, not_keyphr_sim]
        plt.rcParams.update({'font.size': 13})
        # Create a figure instance
        fig = plt.figure(1, figsize=(9, 6))
        # Create an axes instance
        ax = fig.add_subplot(111)
        
        # Create the boxplot
        bp = ax.boxplot(data_to_plot)
        ax.set_xticklabels(['keywords', 'not keywords'])
        plt.title('Cosine Similarity')
        plt.show()
        
        print 'canberra distance'
        keyphr_sim = []
        for f in flattened_kphs:
            if f in candidates:
                keyphr_sim.append(canber[candidates.index(f)])
            
        not_keyphr_sim = []
        for f in candidates:
            if f not in flattened_kphs:
                not_keyphr_sim.append(canber[candidates.index(f)])
        
        data_to_plot = [keyphr_sim, not_keyphr_sim]

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data_to_plot)
        ax.set_xticklabels(['keywords', 'not keywords'])
        plt.title('Canberra Distance')
        plt.show()
#                
#                
#                
        print 'euclidean distance'
        keyphr_sim = []
        for f in flattened_kphs:
            if f in candidates:
                keyphr_sim.append(euclid[candidates.index(f)])
            
        not_keyphr_sim = []
        for f in candidates:
            if f not in flattened_kphs:
                not_keyphr_sim.append(euclid[candidates.index(f)])
        
        data_to_plot = [keyphr_sim, not_keyphr_sim]

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)        
        bp = ax.boxplot(data_to_plot)
        ax.set_xticklabels(['keywords', 'not keywords'])
        plt.title('Euclidean Distance')
        plt.show()
#                
#                print 'sqeuclidean distance'
#                keyphr_sim = []
#                for f in flattened_kphs:
#                    if f in candidates:
#                        keyphr_sim.append(seuclid[candidates.index(f)])
#                    
#                not_keyphr_sim = []
#                for f in candidates:
#                    if f not in flattened_kphs:
#                        not_keyphr_sim.append(seuclid[candidates.index(f)])
#                
#                data_to_plot = [keyphr_sim, not_keyphr_sim]
#        
#                fig = plt.figure(1, figsize=(9, 6))
#                ax = fig.add_subplot(111)        
#                bp = ax.boxplot(data_to_plot)
#                ax.set_xticklabels(['keywords', 'not keywords'])
#                plt.show()
#                
#                print 'correlation'
#                keyphr_sim = []
#                for f in flattened_kphs:
#                    if f in candidates:
#                        keyphr_sim.append(corel[candidates.index(f)])
#                    
#                not_keyphr_sim = []
#                for f in candidates:
#                    if f not in flattened_kphs:
#                        not_keyphr_sim.append(corel[candidates.index(f)])
#                
#                data_to_plot = [keyphr_sim, not_keyphr_sim]
#        
#                fig = plt.figure(1, figsize=(9, 6))
#                ax = fig.add_subplot(111)        
#                bp = ax.boxplot(data_to_plot)
#                ax.set_xticklabels(['keywords', 'not keywords'])
#                plt.show()
#                
#                print 'Scoring candidates...'
#                print 'First score trigrams'
#                

        for tri in ttrigrams:
            token1 = tri[0]
            token2 = tri[1]
            token3 = tri[2]
            score = 0.0
            if token1 in dict_cand_sim.keys():
                score += dict_cand_sim[token1]
            if token2 in dict_cand_sim.keys():
                score += dict_cand_sim[token2]
            if token3 in dict_cand_sim.keys():
                score += dict_cand_sim[token3]  
            dict_cand_sim[token1+' '+token2+' '+token3] = score   
                
        for bi in bbigrams:
            token1 = bi[0]
            token2 = bi[1]
            score = 0.0
            if token1 in dict_cand_sim.keys():
                score += dict_cand_sim[token1]
            if token2 in dict_cand_sim.keys():
                score += dict_cand_sim[token2]
            dict_cand_sim[token1+' '+token2] = score
        
#        dict_cand_sim_final = {}
#        for kc, vc in dict_cand_sim.iteritems():
#            dict_cand_sim_final[ps.stem(kc)] = vc
        
        
        sorted_x = sorted(dict_cand_sim.items(), key=operator.itemgetter(1))
        sorted_x = sorted_x[-_NUM_KEYWORDS:]
        
        print sorted_x
        
        final_set = []
        for sx in sorted_x:
            sxp = sx[0].split(' ')
            local_list = []
            for sxpi in sxp:
                local_list.append(ps.stem(sxpi))
            final_set.append(local_list)
        
        print final_set
        
        
        found = 0
        true_pos = 0
        false_pos = 0
        false_neg = 0      

        for inst_id, inst in enumerate(final_set):
            kphs = vdoc_keys[filev.replace('.txt','.key')]
            flattened_kphs = [val for sublist in kphs for val in sublist]
            flagTP = False
            if len(inst)==1:
                if inst[0] in flattened_kphs:
                    true_pos += 1 
                else:
                    false_pos += 1
            else:
                if len(inst)==2:
                    inst1 = [inst[0],inst[1]]
                    inst2 = [inst[1],inst[0]]
                    for kf in kphs: # check if the candidate exist as a keyphrase bigram
                        flagTP = False
                        if inst[0] in kf and inst[1] in kf:
                            true_pos += 1
                            flagTP = True
                            break 
                    if flagTP==False:# check if the candidate's words exist as a keyword
                        if inst[0] in kphs:
                            true_pos += 1 
                            flagTP = True
                        if inst[1] in kphs:
                            true_pos += 1 
                            flagTP = True
                    if flagTP==False:
                        false_pos += 1
                elif len(inst)==3:
                    for kf in kphs:
                        flagTP = False
                        if inst[0] in kf and inst[1] in kf and inst[2] in kf:
                            true_pos += 1
                            flagTP = True
                            break
                    if flagTP==False:
                        if inst[0] in kphs:
                            true_pos += 1 
                            flagTP = True
                        if inst[1] in kphs:
                            true_pos += 1 
                            flagTP = True
                        if inst[2] in kphs:
                            true_pos += 1 
                            flagTP = True
                    if flagTP==False:
                        for row in kphs:
                            if (inst[0] in row and inst[1] in row and len(row)==2):
                                true_pos += 1 
                                flagTP = True
                            if (inst[0] in row and inst[2] in row and len(row)==2):
                                true_pos += 1 
                                flagTP = True
                            if (inst[2] in row and inst[1] in row and len(row)==2):
                                true_pos += 1 
                                flagTP = True    
                    if flagTP==False:
                        false_pos += 1                  
            
        for inst_id, inst in enumerate(vdoc_keys[filev.replace('.txt','.key')]):
            flattened = [val for sublist in final_set for val in sublist]
            flagTP = False
            if len(inst)==1:
                if inst[0] not in flattened:
                    false_neg += 1
            else:
                if len(inst)==2:
                    for kc in final_set:
                        flagTP = False
                        if inst[0] in kc and inst[1] in kc:
                            flagTP = True
                            break
                    if flagTP==False:
                        false_neg += 1
                else:
                    if len(inst)==3:
                        for kc in final_set:
                            flagTP = False
                            if inst[0] in kc and inst[1] in kc and inst[2] in kc:
                                flagTP = True
                                break
                        if flagTP==False:
                            false_neg += 1 
        print '---------vector size', _DIM_VECTOR, 'iterations', _NUM_ITERATIONS, '-----------'
        p=0.0
        if (true_pos+false_pos)>0:
            p = float(true_pos)/(true_pos+false_pos)
        prec_test2.append(p)
        print p
        r=0.0
        if (true_pos+false_neg)>0:
            r = float(true_pos)/(true_pos+false_neg)  
        rec_test2.append(r)
        print r
        f1 = 0.0
        if (p+r)>0:
            f1 = 2.0*p*r/(p+r)
        fm_test2.append(f1)    
        print f1
        
        true_pos_strict = 0
        true_pos_strict = 0
        false_pos_strict = 0
        false_neg_strict = 0 
        flattened_kphs = [val for sublist in vdoc_keys[filev.replace('.txt','.key')] for val in sublist]
        print set(flattened_kphs)
        flattened_cand = [val for sublist in final_set for val in sublist]
        print set(flattened_cand)
        true_pos_strict = len(set(flattened_kphs).intersection(flattened_cand))
        print true_pos_strict
        false_pos_strict = len(set([x for x in flattened_cand if x not in flattened_kphs]))
        print false_pos_strict
        false_neg_strict = len(set([x for x in flattened_kphs if x not in flattened_cand]))
        print false_neg_strict
        prs=0.0
        if (true_pos_strict+false_pos_strict)>0:
            prs = float(true_pos_strict)/(true_pos_strict+false_pos_strict)
        prec_test_strict.append(prs)
        print prs
        
        rs=0.0
        if (true_pos_strict+false_neg_strict)>0:
            rs = float(true_pos_strict)/(true_pos_strict+false_neg_strict)     
        rec_test_strict.append(rs)
        print rs
        
        f1s = 0.0
        if (prs+rs)>0:
            f1s = 2.0*prs*rs/(prs+rs)
        fm_test_strict.append(f1s)
        print f1s
        
        print 'mean precision strict', sum(prec_test_strict) / float(len(prec_test_strict))
        print 'mean recall strict', sum(rec_test_strict) / float(len(rec_test_strict))
        print 'mean f-measure strict', sum(fm_test_strict) / float(len(fm_test_strict))        

            
        print 'mean precision', sum(prec_test2) / float(len(prec_test2))
        print 'mean recall', sum(rec_test2) / float(len(rec_test2))
        print 'mean f-measure', sum(fm_test2) / float(len(fm_test2))   
    
fig_size = [0,0]
fig_size[0] = 18
fig_size[1] = 8    
plt.rcParams["figure.figsize"] = fig_size 
plt.rcParams.update({'font.size': 11})
t = np.arange(len(fm_test2))
s = fm_test2
plt.scatter(t,s)#, color='k', s=25, marker="o")
plt.plot(t, s)
plt.xlabel('Files')
plt.ylabel('f-measure')
plt.xticks(t,labels,rotation=65)
plt.title('F-measure')
plt.legend()
plt.show() 


t = np.arange(len(rec_test2))
s = rec_test2
plt.scatter(t,s)#, color='k', s=25, marker="o")
plt.plot(t, s)
plt.xlabel('Files')
plt.ylabel('Recall')
plt.xticks(t,labels,rotation=65)
plt.title('Recall')
plt.legend()
plt.show() 


t = np.arange(len(prec_test2))
s = prec_test2
plt.scatter(t,s)#, color='k', s=25, marker="o")
plt.plot(t, s)
plt.xlabel('Files')
plt.ylabel('Precision')
plt.xticks(t,labels,rotation=65)
plt.title('Precision')
plt.legend()
plt.show()   
print 'Finally'            
print 'mean precision', sum(prec_test2) / float(len(prec_test2))
print 'mean recall', sum(rec_test2) / float(len(rec_test2))
print 'mean f-measure', sum(fm_test2) / float(len(fm_test2))        
print 'mean precision strict', sum(prec_test_strict) / float(len(prec_test_strict))
print 'mean recall strict', sum(rec_test_strict) / float(len(rec_test_strict))
print 'mean f-measure strict', sum(fm_test_strict) / float(len(fm_test_strict))
