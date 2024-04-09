from transformers import T5Model, T5Tokenizer
import re
import torch
import numpy as np
import gc

#Load Vocabulary and ProtT5-XL-UniRef50 Model
tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False )
model = T5Model.from_pretrained("prot_t5_xl_uniref50")

gc.collect()

#Load model into GPU and switch to inference mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sequence_examples = ["PRTEINO", "SEQWENCE"]
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(ids['input_ids']).to(device)
print(input_ids.shape)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model.encoder(input_ids=input_ids,attention_mask=attention_mask)
embedding_repr = embedding_repr.last_hidden_state

emb_0 = embedding_repr[0,:7] # shape (7 x 1024)
print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")

average_pooled = torch.mean(embedding_repr, dim=2)
average_pooled_squeezed = average_pooled.squeeze()
average_pooled_squeezed = average_pooled_squeezed.long()
embedding_decoded = model.decoder(input_ids = average_pooled_squeezed)
embedding_decoded = embedding_decoded.last_hidden_state
print('shape of embedding_decoded:', embedding_decoded.shape)

# embedding_decoded = torch.mean(embedding_decoded, dim=2)
# embedding_decoded = embedding_decoded.squeeze()
# embedding_decoded = embedding_decoded.long()

real_lengths = []




decoded = tokenizer.decode(embedding_decoded[0], skip_special_tokens=True)

print(decoded)
