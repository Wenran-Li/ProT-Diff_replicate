from transformers import T5Model, T5Tokenizer
import re
import torch
import numpy as np
import gc

#Load Vocabulary and ProtT5-XL-UniRef50 Model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50")

gc.collect()

#Load model into GPU and switch to inference mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

features = np.load('emb.npy', allow_pickle=True)
feature = torch.from_numpy(features)
features = torch.tensor(features).long()
embedding_decoded = model.decoder(input_ids=features)


decoded = tokenizer.decode(embedding_decoded[0], skip_special_tokens=True)

print(decoded)

