from utils import preprocess_sentence
from tqdm import tqdm
import pickle

class Vocabulary:
    """
    Class to store language vocabulary
    """
    def __init__(self):
        
        # Define special symbols       
        self.word_2_idx = {
            "<SOS>": 0, # Start of string
            "<EOS>": 1, # End of string
            "<PAD>": 2, # Padding tocken
            "<UNK>": 3  # Unknown tocken
        }
        self.idx_2_word = {v: k for k, v in self.word_2_idx.items()}
        
    def index_words(self, tokens):
        """
        Add unique words to the index
        Args:
            tokens (list): The list of tokens, new ones will be indexed. May repeat.
        """
        for token in tokens:
            self.index_word(token)
    
    def index_word(self, token):
        """
        Add unique word to the index
        Args:
            token (str): The token to index. Will be remembered if it is new
        """
        if token not in self.word_2_idx:
            idx = len(self.word_2_idx)
            self.word_2_idx[token] = idx
            self.idx_2_word[idx] = token
    
    def __len__(self):
        """
        Get the number of indexed tokens.
        """
        return len(self.word_2_idx)
    
    def parse_file(self, file, language="english"):
        with open(file, "r") as input_file:
            for line in tqdm(input_file.readlines()):
                tockens = preprocess_sentence(line[:-1], language=language)
                self.index_words(tockens)
    
    @classmethod
    def load(cls, filename):
        vocab = cls()
        with open(filename, "rb") as f:
            all_data = pickle.load(f)
            vocab.word_2_idx = all_data["w2i"]
            vocab.idx_2_word = all_data["i2w"]
        return vocab
        
    def dump(self, filename):
        all_data = {
            "w2i": self.word_2_idx,
            "i2w": self.idx_2_word
        }
        with open(filename, "wb") as f:
            pickle.dump(all_data, f)