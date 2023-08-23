import gensim.downloader as api
import spacy

model = api.load("word2vec-google-news-300")
nlp = spacy.load('en_core_web_sm')

def get_phrase_embedding(phrase):
    words = phrase.split()
    embeddings = [model[word] for word in words if word in model]
    for word in words:
        if word in model:
            print(model[word])
    
    #embeddings = model(words[0])
    #print(embeddings)
    if not embeddings:
        print(f"No embeddings found for any of the words in '{phrase}'")
        return None

    avg_embedding = sum(embeddings) / len(embeddings)
    return avg_embedding

def find_similar_words(word, topn=3):
    try:
        if ' ' in word:
            embedding = get_phrase_embedding(word)
            if embedding is not None:
                similar_words = model.similar_by_vector(embedding, topn=topn)
                return [word[0] for word in similar_words]
            else:
                return []

        else:
            similar_words = model.most_similar(positive=[word], topn=topn)
            return [word[0] for word in similar_words]
    except KeyError:
        print(f"'{word}' not in vocabulary")
        return []

def get_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases

words = get_noun_phrases("segmentation is the best in computer vision")
similar_words_for_phrases = [find_similar_words(word) for word in words]
print(similar_words_for_phrases)
