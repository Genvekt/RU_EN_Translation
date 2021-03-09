from torch.utils.data import Dataset
import pickle
import torch
from data_loaders.vocabulary import Vocabulary

class TextDataset(Dataset):
    def __init__(self, 
                 ru_tokens,
                 ru_voc,
                 en_tokens=None, 
                 en_voc=None,
                 random_sub=0.0,
                 ):
        super().__init__()
        self.random_sub = random_sub
        with open(ru_tokens, "rb") as f:
            self.ru = pickle.load(f)
        self.ru_voc = Vocabulary.load(ru_voc)
        if en_tokens is not None: 
            with open(en_tokens, "rb") as f:
                self.en = pickle.load(f)
            self.en_voc = Vocabulary.load(en_voc)
        else:
            self.en = []
            self.en_voc = Vocabulary()

    def __len__(self):
        return len(self.ru)
    
    def prepare_sequence(self, sequence, voc):
        prepared = []
        for t in sequence:
            # Randomly substitute some tockens with unknown
            if torch.rand(1)[0] < self.random_sub:
                prepared.append(voc.word_2_idx["<UNK>"])
                
            # Substitute tocken with index
            elif t in voc.word_2_idx:
                prepared.append(voc.word_2_idx[t])
                
            # Substitute missing tocken with index of unknown
            else:
                prepared.append(voc.word_2_idx["<UNK>"])
        return torch.tensor(prepared, dtype=torch.uint8)
    
    def sort(self):
        if len(self.en) == len(self.ru):
            data = zip(self.ru, self.en)
            data = sorted(data, key=lambda x: len(x[0]))
            
            self.ru, self.en = zip(*data)
        else:
            pass
    def __getitem__(self, idx):
        ru_item = self.prepare_sequence(self.ru[idx], self.ru_voc)
        if self.en is not None:
            en_item = self.prepare_sequence(self.en[idx], self.en_voc)
            return ru_item, en_item
        
        return ru_item