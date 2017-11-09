from os import listdir
from nltk.stem import PorterStemmer
import operator
import string
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
import numpy as np
from subprocess import call


_STOP_WORDS = [
'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 
'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although',
'always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another',
'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',
'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been', 
'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 
'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 
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

stop = set(stopwords.words('english'))

_WORD_MIN_LENGTH = 3
_WORD_MAX_LENGTH = 35
_NUM_ITERATIONS = 50
_DIM_VECTOR = 50
_FREQUENCY = 1

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


data_original_path = '/home/eirini/Projects_Atypon/NLTKTutorial/SemEval2010-Maui/original/SemEval2010/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin2009/all_docs_abstacts_refined/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/SemEval2010-Maui/original/SemEval2010/'

abstracts_path = '/home/eirini/Projects_Atypon/NLTKTutorial/Semeval2010_Abstracts/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Semeval2010_Abstracts/'
#'/home/eirini/Projects_Atypon/NLTKTutorial/Krapivin_Abstracts/'

print 'Read data files...'
data_original_files = [f for f in listdir(data_original_path)]
abstract_files = [f for f in listdir(abstracts_path)]

ps = PorterStemmer()
vdoc_keys = {}   
prec_test_strict = []
rec_test_strict = []
fm_test_strict = [] 
counter_of_files = 0
for fv, filev in enumerate(data_original_files):
   
    keyphrases = []
    if filev.endswith(".txt") and filev.replace('.txt','.abstr') in abstract_files:
        
        print '###############################################################################'
        counter_of_files += 1
        print 'File No.', counter_of_files, 'Filename', filev.replace('.key','.txt'), '...'
        print 'Word vectors production...'
        call(["./demo.sh", data_original_path+filev, filev, str(_DIM_VECTOR), str(_NUM_ITERATIONS)])
        
        print 'Loading word vectors...'
        W, vocab, ivocab = generate("vocab.txt"+filev.replace('.key','.txt')+str(_DIM_VECTOR)+str(_NUM_ITERATIONS), "vectors"+filev.replace('.key','.txt')+str(_DIM_VECTOR)+str(_NUM_ITERATIONS)+'.txt')

        print 'Find gold keyphrases for file', filev, '... (it is required for the evaluation phase)'
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
        
        dict_word_pos = {}
        with open(abstracts_path+filev.replace('.txt','.abstr'), 'r') as myfile:
            text = myfile.read()
            lowers = filter(lambda y: y in string.printable, text).lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            temp = no_punctuation.split(' ')
            tempn = nltk.word_tokenize(no_punctuation)
            
          
        doc_unigrams = []
        tokens = []
        bbigrams = []
        ttrigrams = []
        no_punctuation_unstemmed = ''
        
        print 'Find all candidate unigrams, bigrams and trigrams for',filev.replace('.key','.txt')     
        for x in tempn:
            no_punctuation_unstemmed += filter(lambda y: y in string.printable, x)+' '
            tokens.append(filter(lambda y: y in string.printable, x))
        for token in tokens:
            token = token.strip().lower()
            token = token.strip(string.digits)
            if len(token) >= _WORD_MIN_LENGTH and len(token) <= _WORD_MAX_LENGTH and '!' not in token and '@' not in token and '#' not in token and '$' not in token and '*' not in token and '=' not in token and '+' not in token and '\\x' not in token and '.' not in token and ',' not in token and '?' not in token and '>' not in token and '<' not in token and '&' not in token and not token.isdigit() and token not in _STOP_WORDS and token not in stop and '(' not in token and ')' not in token and '[' not in token and ']' not in token and '{' not in token and '}' not in token and '|' not in token and token not in doc_unigrams:                                                   
                doc_unigrams.append(token)
        print 'number of unigrams', len(doc_unigrams)
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
        print 'number of trigrams', len(ttrigrams)
        
        _NUM_KEYWORDS = int(len(set(temp))/3)
        count_words = 0
        final_doc_vector = np.zeros((_DIM_VECTOR))
        word_vector = np.zeros((_DIM_VECTOR))
        for word in tempn:
            #print word
            if word in vocab and word in doc_unigrams:
                word_vector = W[vocab[word], :]
                final_doc_vector += word_vector
                count_words += 1 
              
                
        final_doc_vector /= count_words 

        print 'Calculation of cosine similarity between mean_vec and every word - Scoring unigrams...'
        dict_cand_sim = {}
        for word in tempn:
            #print word
            if word in vocab and word in doc_unigrams:
                word_vector = W[vocab[word], :]  
                dict_cand_sim[str(word)] = np.dot(final_doc_vector, word_vector)/(np.linalg.norm(final_doc_vector)* np.linalg.norm(word_vector))

                      
        print 'Scoring bigrams and trigrams...'
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
            dict_cand_sim[str(token1+' '+token2+' '+token3)] = score   
                
        for bi in bbigrams:
            token1 = bi[0]
            token2 = bi[1]
            score = 0.0
            if token1 in dict_cand_sim.keys():
                score += dict_cand_sim[token1]
            if token2 in dict_cand_sim.keys():
                score += dict_cand_sim[token2]
            dict_cand_sim[str(token1+' '+token2)] = score
               
        sorted_x = sorted(dict_cand_sim.items(), key=operator.itemgetter(1))
        sorted_x = sorted_x[-_NUM_KEYWORDS:]
        
        print 'Top', _NUM_KEYWORDS, 'candidate keyphrases with their scores'
        print sorted_x
        
        final_set = []
        for sx in sorted_x:
            sxp = sx[0].split(' ')
            local_list = []
            for sxpi in sxp:
                local_list.append(str(ps.stem(sxpi)))
            final_set.append(local_list)
        print "System's output:"    
        print final_set
        
        print 'Evaluation process'
        kphs = vdoc_keys[filev.replace('.txt','.key')]
        flattened_kphs = [val for sublist in kphs for val in sublist]    
        
        true_pos_strict = 0
        true_pos_strict = 0
        false_pos_strict = 0
        false_neg_strict = 0 
        flattened_kphs = [val for sublist in vdoc_keys[filev.replace('.txt','.key')] for val in sublist]
        print 'Set of gold keywords:'
        print set(flattened_kphs)
        
        print "Set of candidates' keywords:"
        flattened_cand = [val for sublist in final_set for val in sublist]
        print set(flattened_cand)
        
        true_pos_strict = len(set(flattened_kphs).intersection(flattened_cand))
        #print true_pos_strict
        false_pos_strict = len(set([x for x in flattened_cand if x not in flattened_kphs]))
        #print false_pos_strict
        false_neg_strict = len(set([x for x in flattened_kphs if x not in flattened_cand]))
        #print false_neg_strict
        prs=0.0
        if (true_pos_strict+false_pos_strict)>0:
            prs = float(true_pos_strict)/(true_pos_strict+false_pos_strict)
        prec_test_strict.append(prs)
        #print prs
        
        rs=0.0
        if (true_pos_strict+false_neg_strict)>0:
            rs = float(true_pos_strict)/(true_pos_strict+false_neg_strict)     
        rec_test_strict.append(rs)
        #print rs
        
        f1s = 0.0
        if (prs+rs)>0:
            f1s = 2.0*prs*rs/(prs+rs)
        fm_test_strict.append(f1s)
        #print f1s
      
        print '---------------------------------------------------------------------------'
        print 'mean precision strict', sum(prec_test_strict) / float(len(prec_test_strict))
        print 'mean recall strict', sum(rec_test_strict) / float(len(rec_test_strict))
        print 'mean f-measure strict', sum(fm_test_strict) / float(len(fm_test_strict))        
    break        

