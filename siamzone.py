from bs4 import BeautifulSoup
from pythainlp import word_tokenize, corpus
from gensim.models import word2vec
import requests
import collections
import json
import numpy as np
import csv

url = 'https://www.siamzone.com/music/thailyric/' # + id 5 digits

def scrape(start_id, end_id):
    with open('siamzone{}-{}.json'.format(start_id, end_id), 'w') as f:
        all_dic = {}
        for id in range(start_id, end_id):
            response = requests.get(url + str(id))
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                lyrics = soup.find('div', id="lyrics-content").text.split('\n\n', 1)[-1].strip('\r\n').strip('\t').strip('\n')
                date = soup.find('meta', itemprop="datePublished").get('content')

                try:
                    title_artist = soup.find('meta', property="og:title").get('content').split('-')
                    title = title_artist[0].split(' ', 1)[-1].strip()
                    artist = title_artist[1].strip()
                    all_dic[id] = {'artist':artist, 'title':title, 'lyrics':lyrics, 'date':date}
                except:
                    print(id)

                
            else:
                pass
        json.dump(all_dic, f, indent=4, ensure_ascii=False)

def tokenize(lyrics):
    seqs = lyrics.split('\n')
    tokens = [word_tokenize(seq, keep_whitespace=False) for seq in seqs]
    tokens = [seq for seq in tokens if seq != []]
    return tokens

def tokenize_id(lyrics, id):
    seqs = lyrics.split('\n')
    tokens = [word_tokenize(seq, keep_whitespace=False) for seq in seqs]
    tokens = [[id]+seq for seq in tokens if seq != []]  # [[id, token1, token2,....], [id, ],...] list of each line
    return tokens

def make_txt():
    with open('siamzone.json', 'r') as f:
        data = json.load(f)
    with open('sizamzone2.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ', lineterminator='\n')
        ids = data.keys()
        for id in ids:
            lyr = data[id]['lyrics']
            writer.writerows(lyr)
    """
    with open('siamzone_token.json', 'w') as f:
        ids = data.keys()
        dic = {}
        for id in ids:
            lyr = data[id]['lyrics']
            dic[id] = tokenize(lyr)
        json.dump(dic, f, indent=4, ensure_ascii=False)
    """

def make_model(skip_gram=True, epoch=5):
    with open('siamzone.txt', 'r') as f:
        if skip_gram:
            skip = 1
            savename = 'siamzone_skip'
        else:
            skip = 0
            savename = 'siamzone_cbow'
        sentences = word2vec.LineSentence(f)
        model = word2vec.Word2Vec(sentences, sg=skip, size=300, min_count=3, window=5, iter=epoch)
        #model.save(savename + '.model')
        model.wv.save_word2vec_format(savename + '.bin', binary=True)

class SiamZone:
    def __init__(self):
        with open('siamzone.txt', 'r') as f:
            self.lines = f.read().split('\n')[:-1]  # [[id, token1, token2,....], [],...]
        self.dic = {}

    def word_freq(self):
        count = collections.Counter()
        for line in self.lines:
            tokens = line.split(' ')[1:]  # exclude id 
            id = line.split(' ')[0]
            for token in tokens:
                count[token] += 1
        return count

    def ngram(self, n=2):
        count = collections.Counter()
        for line in self.lines:
            tokens = line.split(' ')[1:]  # exclude id 
            for i in range(len(tokens)-n+1):
                count[tuple(tokens[i:i+n])] += 1
        return count

    def make_tfidf(self):
        self.tf_dic = {}
        self.idf_dic = {}
        for line in self.lines:
            if len(line.split()) > 1:
                document = line.split()[0]
                tokens = line.split()[1:]
                if document not in self.tf_dic:
                    self.tf_dic[document] = {}
                for token in tokens:
                    self.tf_dic[document][token] = self.tf_dic[document].get(token, 0) + 1
                    if token not in self.idf_dic:
                        self.idf_dic[token] = set(document)
                    else:
                        self.idf_dic[token].add(document)
        self.N = len(self.tf_dic.keys())

    def tf_idf(self, word, document):
        document = str(document)
        tf = 1 + np.log10(self.tf_dic[document][word])
        idf = np.log10(self.N / len(self.idf_dic[word]))
        return tf * idf

    def tfidf_ranking(self, document, n=10):
        document = str(document)
        tokens = self.tf_dic[document].keys()
        counter = collections.Counter()
        for token in tokens:
            counter[token] = self.tf_idf(token, document)
        return counter.most_common()[:n], counter.most_common()[::-1][:n]

    def one_word(self, word):
        documents = self.tf_dic.keys()
        counter = collections.Counter()
        for document in documents:
            try:
                counter[document] = self.tf_idf(word, document)
            except:
                pass
        values = np.array(list(dict(counter).values()))
        print('mean', np.mean(values))
        print('max', np.max(values))
        print('min', np.min(values))
        print('median', np.median(values))
        print('sd', np.std(values))
        return counter

sz = SiamZone()
sz.make_tfidf()