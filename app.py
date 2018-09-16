from flask import Flask, render_template

import nltk
import textblob as txb
import pandas as pd
import numpy as np
import re
import pickle

'''
config is a global dictionary which holds 
some specific precalculated stuff for optimization purposes
'''
config = {
    'ps' : nltk.PorterStemmer(),
    'inp_q_features' : None,
    'inp_q_tokens' : None,
    'inp_q' : None, 
    'helping_word' : ['am', 'are', 'is', 'was', 'were', 'be', 'being', 'been','have', 'has', 'had', 'shall', 'will','do', 'does', 'did', 'may', 'must', 'might', 'can', 'could', 'would', 'should', 'i'],
    'clusters' : None,
    'len_nulls' : None,
    'largest_cluster' : None,
    'smallest_cluster' : None
}

'''
This function first converts the input sentence to lower case
Then it removes all the slangs which are 's, 'll ? etc and returns
'''
def preprocessing(sentence):
    review_text = sentence.lower()
    #review_text = re.sub(r"[A-Za-z0-9]", " ", review_text)
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`\"]", " ", review_text)
    review_text = re.sub(r"\'s", " ", review_text)
    review_text = re.sub(r"\'ve", " ", review_text)
    review_text = re.sub(r"n\'t", " ", review_text)
    review_text = re.sub(r"\'re", " ", review_text)
    review_text = re.sub(r"\'d", " ", review_text)
    review_text = re.sub(r"\'ll", " ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\'", "", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    return review_text

'''
This function takes a sentence as input and returns all the stemmed nouns
'''
def get_nouns(text):
    ps = config['ps']
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    #getting all the nouns
    nouns = [i for i in tagged if i[1][0]=='N']
    f_nouns = []
    for i in nouns:
        '''if(i[1]=='NNS'):#if it is common noun in plural form then get its stemmed word
            f_nouns.append(ps.stem(i[0]))
        else:
            f_nouns.append(i[0].lower())'''
        f_nouns.append(ps.stem(i[0]))
    return set(f_nouns)

'''
This function returns the noun phrases for a given input sentence
'''
def get_noun_phrases(sentence):
    wiki = txb.TextBlob(sentence)
    noun_phrases = [str(i) for i in wiki.noun_phrases]
    return set(noun_phrases)

'''
This function returns the verbs, adverbs
'''
def get_verbs_with_addons(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return set([i[0] for i in tagged if i[1][0] in ('R','V') and i[0] not in config['helping_word']])

'''
This function takes the input as sentence
It does preprocessing of the sentence
Then it finds out the nouns, noun phrases, verbs and adverbs
and returns the set of all these features
'''
def get_features(sentence):
    #removing redundancy by eliminating all the nouns which are already present in the noun phrase
    text = preprocessing(sentence)
    
    nouns = get_nouns(text)
    noun_phrases = get_noun_phrases(text)
    verbs_adverbs = get_verbs_with_addons(text)
    
    #if both the sets are empty then return a null set else if one is empty then return other
    if((len(nouns)==0) and (len(noun_phrases)==0)):
        return verbs_adverbs
    if(len(nouns)==0):
        return noun_phrases.union(verbs_adverbs)
    if(len(noun_phrases)==0):
        return nouns.union(verbs_adverbs)
    #if both non empty then select just the uniques
    nn = [[i,0] for i in nouns]
    for nph in noun_phrases:
        for n in nn:
            if(n[0] in nph):
                n[1]=1
    x=[]
    for i in nn:
        if(i[1]==0):
            x.append(i[0])

    x = [i[0] for i in nn if i[1]==0]
    return set(x+list(noun_phrases)+list(verbs_adverbs))

'''
construct a reduced set of questions by 
taking the union of the clusters of the keywords present 
in the input questions
'''
def construct_reduced_question_pool(clusters, state):
    tokens_present = []
    if(state==True):
        q_tokens = list(config['inp_q_features'])
        f_list = list()
        for i in q_tokens:
            try:
                f_list = f_list + clusters[i]
                tokens_present.append(i)
            except:
                pass
        f_list = list(set(f_list))
        return f_list, tokens_present
    else:
        #if the input question did not give us features then check for it in the common null pool
        return list(clusters['-1']), None

#this function returns the best 5 matching questions from among the dataset
def rank(indexed_features):
    q_tokens = config['inp_q_tokens']
    results = list()
    for i in indexed_features.index:
        sim_ind = jaccard(q_tokens, indexed_features.loc[i,'features'])
        results.append({'sim_index':sim_ind,'question':indexed_features.loc[i,'question']})
    results.sort(key = lambda x:x['sim_index'], reverse=True)
    #return results[0:5]
    return [i['question'] for i in results][0:10]
    
'''
This function takes a question as input and finds out the features of it
then it splits every of those features and returns it as a list of keywords
'''
def tokenized_features(inp_ques):
    t_features = []
    features = get_features(inp_ques)
    for i in features:
        t_features = t_features + [j for j in i.split()]
    return set(t_features)

'''
takes 2 sets of tokens and finds out it jaccard similarity
'''
def jaccard(tokens1, tokens2):
    #receives the set of tokens1 and token2
    inter = tokens1.intersection(tokens2)
    uni = tokens1.union(tokens2)
    return float(len(inter)/len(uni))

'''
Load the clusters saved as a pickle file
'''
def load_clusters(name='clusters.pkl'):
    with open(name, 'rb') as f:
        clusters = pickle.load(f)
        config['clusters'] = sum([1 for i in clusters.keys()])
        config['len_nulls'] = len(clusters['-1'])
        tmp = [len(clusters[i]) for i in clusters.keys()]
        config['largest_cluster'] = max(tmp)
        config['smallest_cluster'] = min(tmp)
        return clusters

#returns True if there are any features of the input question
#returns False if there are no features of the input question
def init_question(question):
    question = preprocessing(question)
    config['inp_q'] = question
    config['inp_q_features'] = get_features(question)
    config['inp_q_tokens'] = tokenized_features(question)
    if(config['inp_q_features'] is None):
        return False
    else:
        return True

def init_system():
    clusters = load_clusters()
    df = pd.read_csv('featured_questions.csv')
    return clusters, df
    
'''
This function takes a question as input and returns all the related questions
'''
def answer_workflow(question):
    val = init_question(question)
    questions_list, tokens_found = construct_reduced_question_pool(clusters, val)
    reduced_with_features = df.iloc[[i for i in questions_list],:]
    result = rank(reduced_with_features)
    return result, tokens_found


app = Flask(__name__)
#loading up the clusters and dataframe
clusters, df = init_system()

@app.route('/')
def index():
   return render_template('index.html')


@app.route('/ask/<question>')
def get_results(question):
    
    list_answers, tokens_found = answer_workflow(question)        
    result = { 'tokens' : tokens_found, 'answers' : list_answers }
    return render_template('answers.html', answers = result)
'''
@app.route('/ask/<question>')
def get_results(question):
    
    list_answers, tokens_found = answer_workflow(question)   
    return render_template('answers.html', answers = list_answers)
'''

#This can be used to show the user the details of the working application
@app.route('/details')
def show_app_details():
    return render_template('details.html', details = config)

if __name__ == '__main__':
   app.run(debug = True)