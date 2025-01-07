import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval


'''Create data loader'''

class MovieDataset(Dataset):
    def __init__(self, filename):
        self.df = pd.read_csv(filename, converters={'input_x': literal_eval})
        
        # print(self.df['input_x'])

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        ## load the input features and labels
        ##-----------------------------------------------
        ## complete the code to load features and labels
        ##-----------------------------------------------
        #by this point: pre-processing has made indices
        #in the preprocessed csv file have columns: content, label, seq_len, clean_text, seq_words, input_x
         # Load input features from the "input_x" column at the specified index
        input_x = self.df.loc[index, 'input_x']
        
#TEST
    # Load label from the "Label" column at the specified index
        label = self.df.loc[index, 'Label']

        

        

        return torch.tensor(input_x), torch.tensor(label,dtype=torch.float)
        
        