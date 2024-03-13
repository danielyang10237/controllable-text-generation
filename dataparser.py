import nltk
import os
if not nltk.download('stopwords'):
    nltk.download('stopwords')
if not nltk.download("wordnet"):
    nltk.download("wordnet")
from nltk.corpus import wordnet as wn
import random

from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))
english_stopwords.add('the')
line_index = 0

# check if datasets/target.txt does not exist
if not os.path.exists('datasets/target.txt') or True:
    with open('datasets/wikisent2.txt', 'r') as preprocessed_data, open('datasets/target.txt', 'w') as t, open('datasets/scrambled.txt', 'w') as f:

            for line in preprocessed_data:
                if line_index % 400 == 0:
                    words = line.split()

                    # remove stopwords from the sentence
                    words = [word for word in words if word not in english_stopwords]

                    # reshuffle the word 20% of the time
                    for i, word in enumerate(words):
                        if random.random() < 0.09:
                            synsets = wn.synsets(word)
                            if synsets:
                                words[i] = synsets[0].lemmas()[0].name()
                        elif random.random() < 0.09:
                            # shuffle the word
                            j = random.randint(0, len(words) - 1)
                            words[i], words[j] = words[j], words[i]
                
                    presentence = ' '.join(words)
                    f.write(presentence + '\n')
                
                    t.write(line)
                line_index += 1
    
