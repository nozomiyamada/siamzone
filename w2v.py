import json
import re
import os
import csv
import pandas as pd
import random
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import collections
import numpy as np
from glob import glob
from pythainlp import word_tokenize
from gensim.models import word2vec
from gensim.models import KeyedVectors

def make_model(text_file='/Users/Nozomi/files/news/thairath/json/cat.txt', skipgram=0, epoch=3):
    if skipgram == 0:
        save_name = text_file.rsplit('/', 1)[0] + '/cbow'
    else:
        save_name = text_file.rsplit('/', 1)[0] + '/skip'
    sentences = word2vec.LineSentence(text_file)
    model = word2vec.Word2Vec(sentences, sg=skipgram, size=300, min_count=5, window=5, iter=epoch)  # CBOW: sg=0, skip-gram: sg=1
    model.save(save_name+'.model')
    model.wv.save_word2vec_format(save_name+'.bin', binary=True)

def cos_sim(v1, v2):
    return round(float(np.dot(v1, v2)) / (norm(v1) * norm(v2)), 4)

def norm(vector):
    return round(np.linalg.norm(vector), 4)

class Word2Vector:
    def __init__(self):
        self.model = None
        self.vocab = []

    # one word > vector
    def vec(self, word):
        return self.model.wv[word]

WV = Word2Vector()  # instance

def load_model(skipgram=False):
    if skipgram:
        WV.model = word2vec.Word2Vec.load('/Users/Nozomi/files/news/thairath/json/skip.model')
    else:
        WV.model = word2vec.Word2Vec.load('/Users/Nozomi/files/news/thairath/json/cbow.model')
    WV.vocab = list(WV.model.wv.vocab.keys())

# search most similar n words
def sim(word, n=10):
    results = WV.model.wv.most_similar(positive=[word], topn=n)
    for result in results:
        print(result[0], round(result[1], 4))

def mahalanobis(vectors):
    mean_vec = np.mean(vectors, axis=0)
    deviation_vec = vectors - mean_vec
    cov_matrix = np.cov(vectors.T, bias=False)
    inv_matrix = np.linalg.inv(cov_matrix)
    mahal_dis = list(map(lambda vec: np.sqrt(np.dot(np.dot(vec, inv_matrix), vec.T)), deviation_vec))
    return mahal_dis

# calculate similarity & distance of two words
def sim_two(word1, word2):
    return cos_sim(WV.vec(word1), WV.vec(word2))


####### OLD CODE ###############################

class Metonymy:

    def __init__(self, model):
        self.model = word2vec.Word2Vec.load(model)
        self.vocab = list(self.model.wv.vocab.keys())
    
    # return similarity of random two words
    def sim_two_word_random(self):
        words = random.sample(self.vocab, 2)
        #while words[0][0].isalpha() == False or words[1][0].isalpha() == False:
            #words = random.sample(self.vocab, 2)
        #print(words)
        return (cos_sim(self.vec(words[0]),self.vec(words[1])))
    
    # plot distributions of similarity
    def sim_random(self, k, log=False):
        sim_list = [self.sim_two_word_random() for i in range(k)]

        
        """
        x = np.linspace(-1,1,201)
        param = norm.fit(sim_list)
        pdf_fitted = norm.pdf(x,loc=param[0], scale=param[1])
        pdf = norm.pdf(x)
        print(param)
        
        plt.figure
        plt.title('Normal distribution')
        plt.plot(x, pdf_fitted, 'r-')
        """
        
        ret = plt.hist(sim_list, bins=200, range=(-1,1))
        plt.xlabel('cosine similarity')
        
        if log == True:
            plt.yscale('log')
        plt.title('cosine similarity of random {} word pairs (bin=0.01)'.format(k))
        plt.show()
        new_list = [0]
        for i in ret[0][1:]:
            new_list.append(new_list[-1]+i)
        for i,j in zip(ret[1],new_list):
            print(i,(k-j)/k*100)
    
    # return distance of random two words
    def dis_pair_random(self):
        words = random.sample(self.vocab, 2)
        #while words[0][0].isalpha() == False or words[1][0].isalpha() == False:
            #words = random.sample(self.vocab, 2)
        #print(words)
        return (np.linalg.norm(self.model.wv[words[0]] - self.model.wv[words[1]]))
    
    # plot distributions of distance of random pairs
    def dis_pairs(self, num_of_pair, log=False):
        dis_list = [self.dis_pair_random() for i in range(num_of_pair)]
        
        plt.hist(dis_list, bins=240, range=(0,60))
        plt.xlabel('Euclidean distance')
        plt.ylabel('numbers')
        if log == True:
            plt.yscale('log')
        plt.title('distances of random {} word pairs (bin=0.25)'.format(num_of_pair))
        plt.show()
    
    # plot distance of random k words
    def dis_words(self, k):
        words = random.sample(self.vocab, k)
        dis_list = [norm(self.vec(word)) for word in words]
        
        plt.hist(dis_list, bins=240, range=(0,60))
        plt.xlabel('Euclidean distance from zero')
        plt.ylabel('numbers')
        plt.title('distances of random {} words (bin=0.25)'.format(k))
        plt.show()
        
    # vector calculation
    def calc(self, posi, nega, n=10):
        results = self.model.wv.most_similar(positive=posi, negative=nega, topn=n)
        for result in results:
            print(result[0], round(result[1], 4))
    
    def vec_dis(self, open_tsv='metonymy_vector.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        label = [line[0] for line in lines]
        
        vectors = np.array([line[2:] for line in lines],dtype=float)
        dis = np.linalg.norm(vectors, axis=1)
        corr = np.corrcoef(vectors)
        print(np.mean(dis))
        
        plt.barh(np.arange(len(label)), dis, tick_label=label)
        plt.xlabel('Euclidean distance between metonymy and country')
        plt.savefig('test.png', format='png', dpi=150)
        plt.show()
    
    def mahal(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        metonymy_vectors = [self.model.wv[line[0]] for line in lines]
        country_vectors = [self.model.wv[line[1]] for line in lines]
        
        metonymy_mahal = mahalanobis(np.array(metonymy_vectors,dtype=float))
        country_mahal = mahalanobis(np.array(country_vectors,dtype=float))
        
        return metonymy_mahal, country_mahal
    
    # similarity of each metonym vector    
    def table_sim(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        met_vec = [self.model.wv[line[0]] for line in lines]
        label = [line[0] for line in lines]
        label.append('mean')
        
        sims = np.zeros((len(lines), len(lines)+1))
        for i in range(len(lines)):
            for j in range(len(lines)):
                sims[i][j] = round(cos_sim(met_vec[i], met_vec[j]) , 3)
            sims[i][-1] = np.mean(sims[i][:-1])
        
        df = pd.DataFrame(sims, columns=label, index=label[:-1])
        return df
    
    def table_dis(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        met_vec = np.array([self.model.wv[line[0]] for line in lines], dtype=float)
        country_vec = np.array([self.model.wv[line[1]] for line in lines], dtype=float)
        
        table = np.zeros((len(lines), len(lines)+1))
        for i in range(len(lines)):
            for j in range(len(lines)):
                table[i][j] = round(np.linalg.norm(met_vec[i]-met_vec[j]), 3)
            table[i][-1] = np.mean(table[i][:-1])
        label = [line[0] for line in lines]
        label.append('mean')
        df = pd.DataFrame(table, columns=label, index=label[:-1])
        return df
    
    # similarity of each metonymiztion vector    
    def table_sim_metonymize(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        met_vec = [self.model.wv[line[0]] for line in lines]
        country_vec = [self.model.wv[line[1]] for line in lines]
        
        metonymize = np.array([m - c for m, c in zip(met_vec,country_vec)],dtype=float)
        label = [line[0] for line in lines]
        
        sims = np.zeros((len(lines), len(lines)))
        for i in range(len(lines)):
            for j in range(len(lines)):
                sims[i][j] = round(cos_sim(metonymize[i], metonymize[j]) , 3)
        
        df = pd.DataFrame(sims, columns=label, index=label)
        return df

    def affine_diag(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        metonymy_vectors = np.array([self.model.wv[line[0]] for line in lines], dtype=float)
        country_vectors = np.array([self.model.wv[line[1]] for line in lines], dtype=float)
        
        mean_metonymize = np.mean(metonymy_vectors - country_vectors, axis=0)
        
        coef_list = []
        intercept_list = []
        for i in range(300):
            regression = np.polyfit(country_vectors.T[i], metonymy_vectors.T[i],1)
            coef_list.append(regression[0])
            intercept_list.append(regression[1])
        
        self.A_diag = np.diag(coef_list)
        self.b_diag = intercept_list
        print('similarity of metonymy vector and coefficient b', cos_sim(mean_metonymize,intercept_list))
        return np.linalg.det(np.diag(coef_list))
    
    def affine_full(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        
        metonymy_vectors = np.array([self.model.wv[line[0]] for line in lines], dtype=float)
        country_vectors = np.array([self.model.wv[line[1]] for line in lines], dtype=float)
        
        self.mean_met_vec = np.mean(metonymy_vectors - country_vectors, axis=0)
        
        coef_list = []
        intercept_list = []
        for i in range(300):
            y = metonymy_vectors[:,i].T
            lr.fit(country_vectors,y)
            coef_list.append(lr.coef_)
            intercept_list.append(lr.intercept_)
        
        self.A_full = np.array(coef_list)
        self.b_full = intercept_list
        print('similarity of metonymy vector and coefficient b', cos_sim(self.mean_met_vec,intercept_list))
        open_file.close()
        return np.linalg.det(np.array(coef_list))
    
    def apply_affine_diag(self, country):
        map_vec = np.dot(self.A_diag, self.model.wv[country]) + self.b_diag
        return map_vec
    
    def apply_affine_full(self, country):
        map_vec = np.dot(self.A_full, self.model.wv[country]) + self.b_full
        return map_vec
    
    # compare 'only vector' 'diag Affine' 'full Affine'
    def compare_affine(self,country, metonym):
        print('+ mean vec | diag affine | full affine')
        
        a = (cos_sim(self.vec(country)+self.mean_met_vec, self.vec(metonym)))
        b = (cos_sim(self.apply_affine_diag(country), self.vec(metonym)))
        c = (cos_sim(self.apply_affine_full(country), self.vec(metonym)))
        
        print(str(a) +'|'+ str(b) +'|'+ str(c))
    
    # for compare all word pair
    def compare_all(self, open_tsv='wordpair.tsv'):
        with open(open_tsv, 'r', encoding='utf-8') as open_file:
            lines = list(csv.reader(open_file, delimiter='\t'))
        
            for line in lines:
                print(line[1]+' : '+line[0])
                self.compare_affine(line[1],line[0])
            
    def search(self, country, k=10):
        affine = self.apply_affine_diag(country)
        parallel = self.vec(country) + self.mean_met_vec
        
        all_sim_affine = {cos_sim(affine, self.vec(word)):word for word in self.vocab}
        all_sim_parallel = {cos_sim(parallel, self.vec(word)):word for word in self.vocab}
        result_affine = sorted(all_sim_affine.items())[-k:][::-1]
        result_parallel = sorted(all_sim_parallel.items())[-k:][::-1]
        print('Affine Transformation')
        print(result_affine)
        print('\nParallel Translation')
        print(result_parallel)

    def save_embedding_projector(self, open_tsv='wordpair.tsv'):
        open_file = open(open_tsv, 'r', encoding='utf-8')
        vec_file = open('metonymy.tsv', 'w', encoding='utf-8')
        label_file = open('label.tsv', 'w', encoding='utf-8')
        lines = list(csv.reader(open_file, delimiter='\t'))
        for line in lines:
            vec_file.write('\t'.join(map(str,self.model.wv[line[0]]))+'\n')
            label_file.write(line[0] + '\n')
        for line in lines:
            vec_file.write('\t'.join(map(str,self.model.wv[line[1]]))+'\n')
            label_file.write(line[1] + '\n')
        for other in self.vocab[:10000]:
            vec_file.write('\t'.join(map(str,self.vec(other)))+'\n')
            label_file.write(other + '\n')
        
        open_file.close()
        vec_file.close()
        label_file.close()

