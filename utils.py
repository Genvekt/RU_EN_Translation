from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
import csv
import pickle
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def preprocess_sentence(sentence, language):
    return word_tokenize(sentence, language=language)

def text_to_pickle(text_file, file_to_save, language):
    
    all_tokens = []
    
    with open(text_file, "r") as input_file:
        for line in tqdm(input_file.readlines()):
            tockens = preprocess_sentence(line[:-1], language=language)
            all_tokens.append(tockens)
    
    with open(file_to_save, "wb") as f:
        pickle.dump(all_tokens, f)
        
def train_val_split(text_file1,
                    text_file2, 
                    train_file1,
                    train_file2, 
                    val_file1, 
                    val_file2):
    
    with open(text_file1, "rb") as f:
        all_tokens1 = pickle.load(f)
        
    with open(text_file2, "rb") as f:
        all_tokens2 = pickle.load(f)
        
    idx = list(range(len(all_tokens1)))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    with open(train_file1, "wb") as f:
        pickle.dump([all_tokens1[i] for i in train_idx], f)
    
    with open(train_file2, "wb") as f:
        pickle.dump([all_tokens2[i] for i in train_idx], f)
        
        
    with open(val_file1, "wb") as f:
        pickle.dump([all_tokens1[i] for i in val_idx], f)
    
    with open(val_file2, "wb") as f:
        pickle.dump([all_tokens2[i] for i in val_idx], f)
    
   
    