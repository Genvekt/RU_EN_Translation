import random
import torch 

class Batches:
    def __init__(self, dataset, batch_size, sort=True, train=True):
        self.train = train
        self.dataset = dataset
        self.batch_size = batch_size
        self.len = len(dataset) // batch_size
        if self.len * batch_size < len(dataset):
            self.len += 1
            
        if self.train:
            self.dataset.sort()
            
    def iterate(self):
        # Create batch indexes and shuffle them
        batch_ids = list(range(self.len))
        random.shuffle(batch_ids)
        
        for idx in batch_ids:
            start_id = idx*self.batch_size
            end_id = min((idx+1)*self.batch_size, len(self.dataset))
            
            batch_ru = []
            batch_en = []
            # Create ru batch
            for item_idx in range(start_id, end_id):
                items = self.dataset[item_idx]
                if self.train:
                    batch_ru.append(items[0])
                    batch_en.append(items[1])
                else:
                    batch_ru.append(items)
            
            batch_ru =  self.pad_batch(batch_ru, 
                                       self.dataset.ru_voc.word_2_idx['<PAD>'])
            
            if self.train:
                batch_en =  self.pad_batch(batch_en, 
                                           self.dataset.en_voc.word_2_idx['<PAD>'])
                
                yield batch_ru, batch_en
                
            else:
                yield batch_ru
        
        
    def pad_batch(self, batch, pad_idx):
        # Get the length of longest sentence in batch
        max_len = max([len(item) for item in batch])
        
        # Create empty batch
        padded_batch = torch.full((len(batch), max_len) , fill_value=pad_idx)
        for item_id in range(len(batch)):
            item = batch[item_id]
            padded_batch[item_id, :len(item)] = item
        return padded_batch
    
    def pad_idx(self, is_source=True):
        if is_source:
            return self.dataset.ru_voc.word_2_idx['<PAD>']
        else:
            return self.dataset.en_voc.word_2_idx['<PAD>']