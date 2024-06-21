import polaris as po
import mol2vec
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, sentences2vec

# Use Mol2Vec instead of MACCSkeys
model_path = 'model_300dim.pkl'  # Path to the pre-trained model
mol2vec_model = word2vec.Word2Vec.load(model_path)

# print("Managed to run word2vec model import")

benchmark = po.load_benchmark("polaris/pkis1-kit-wt-mut-c-1")
train, test = benchmark.get_train_test_split()
smi = [ x[0] for x in train ]

from rdkit.Chem import Draw
from rdkit import Chem

mols = [ Chem.MolFromSmiles(x) for x in smi ]

import torch
import torch.nn as nn 

act = torch.tensor([ list(x[1].values()) for x in train ])
mask = ~torch.any(torch.isnan(act), axis=1)
act = act[mask]
act.size()

wt_seq = "MRGARGAWDFLCVLLLLLRVQTGSSQPSVSPGEPSPPSIHPGKSDLIVRVGDEIRLLCTDPGFVKWTFEILDETNENKQNEWITEKAEATNTGKYTCTNKHGLSNSIYVFVRDPAKLFLVDRSLYGKEDNDTLVRCPLTDPEVTNYSLKGCQGKPLPKDLRFIPDPKAGIMIKSVKRAYHRLCLHCSVDQEGKSVLSEKFILKVRPAFKAVPVVSVSKASYLLREGEEFTVTCTIKDVSSSVYSTWKRENSQTKLQEKYNSWHHGDFNYERQATLTISSARVNDSGVFMCYANNTFGSANVTTTLEVVDKGFINIFPMINTTVFVNDGENVDLIVEYEAFPKPEHQQWIYMNRTFTDKWEDYPKSENESNIRYVSELHLTRLKGTEGGTYTFLVSNSDVNAAIAFNVYVNTKPEILTYDRLVNGMLQCVAAGFPEPTIDWYFCPGTEQRCSASVLPVDVQTLNSSGPPFGKLVVQSSIDSSAFKHNGTVECKAYNDVGKTSAYFNFAFKGNNKEQIHPHTLFTPLLIGFVIVAGMMCIIVMILTYKYLQKPMYEVQWKVVEEINGNNYVYIDPTQLPYDHKWEFPRNRLSFGKTLGAGAFGKVVEATAYGLIKSDAAMTVAVKMLKPSAHLTEREALMSELKVLSYLGNHMNIVNLLGACTIGGPTLVITEYCCYGDLLNFLRRKRDSFICSKQEDHAEAALYKNLLHSKESSCSDSTNEYMDMKPGVSYVVPTKADKRRSVRIGSYIERDVTPAIMEDDELALDLEDLLSFSYQVAKGMAFLASKNCIHRDLAARNILLTHGRITKICDFGLARDIKNDSNYVVKGNARLPVKWMAPESIFNCVYTFESDVWSYGIFLWELFSLGSSPYPGMPVDSKFYKMIKEGFRMLSPEHAPAEMYDIMKTCWDADPLKRPTFKQIVQLIEKQISESTNHIYSNLANCSPNRQKPVVDHSVRINSVGSTASSSQPLLVHDDV"
mut1_seq = wt_seq[:559] + 'G' + wt_seq[559 + 1:]
mut2_seq = wt_seq[:669] + 'I' + wt_seq[669 + 1:]
import os.path



fname = "seq_embeddings.torch"
if os.path.isfile(fname):
    token_representations = torch.load(fname)

else:

    import torch
    import esm

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("wt", wt_seq),
        ("mut1", mut1_seq),
        ("mut2", mut2_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    if not os.path.isfile(fname):
        torch.save(token_representations, fname)

print(token_representations.size())

import matplotlib.pyplot as plt

plt.plot((token_representations[0] - token_representations[1]).abs().sum(dim=1))
plt.plot((token_representations[0] - token_representations[2]).abs().sum(dim=1))
# plt.show()

from rdkit.Chem import MACCSkeys

wt = [token_representations[0][559], token_representations[0][669]]
ti670 = [token_representations[2][559], token_representations[2][669]]
vg560 = [token_representations[1][559], token_representations[1][669]]

print((wt[0] - ti670[0]).abs().sum(), (wt[1] - ti670[1]).abs().sum())

new_train = []

for entry in train:
    smiles = entry[0]
    wt_act = entry[1]['CLASS_KIT']
    vg560_act = entry[1]['CLASS_KIT_(V560G_mutant)']
    ti670_act = entry[1]['CLASS_KIT_(T6701_mutant)']

    mol = Chem.MolFromSmiles(smiles) # keep this
    # print("Check mol2vec embedding shape")

    sentence = mol2alt_sentence(mol, radius=1) # I MIGHT need to convert to list before next line
    # print("What is the length of sentence?", len(sentence))
    
    #mol2vec_embedding shape is (1, 300)
    mol2vec_embedding = sentences2vec([sentence], mol2vec_model, unseen='UNK')
    reshaped_mol2vec_embedding = mol2vec_embedding.reshape(300)
    # print("what is shape of reshaped mol2vec?", reshaped_mol2vec_embedding.shape)

    # maccs = torch.tensor(list(MACCSkeys.GenMACCSKeys(mol))).double() # modify this
    # print("What is the shape a maccs embedding? ", maccs.shape)

    # if wt_act in [0, 1]:
    #     new_train.append( ([ maccs, wt[0], wt[1] ], wt_act) )
    # if ti670_act in [0, 1]:
    #     new_train.append( ([ maccs, ti670[0], ti670[1] ], ti670_act) )
    # if vg560_act in [0, 1]:
    #     new_train.append( ([ maccs, vg560[0], vg560[1] ], vg560_act) )
    
    if wt_act in [0, 1]:
        new_train.append( ([ torch.tensor(reshaped_mol2vec_embedding).double(), wt[0], wt[1] ], wt_act) )
    if ti670_act in [0, 1]:
        new_train.append( ([ torch.tensor(reshaped_mol2vec_embedding).double(), ti670[0], ti670[1] ], ti670_act) )
    if vg560_act in [0, 1]:
        new_train.append( ([ torch.tensor(reshaped_mol2vec_embedding).double(), vg560[0], vg560[1] ], vg560_act) )
    

from torch.utils.data import Dataset

class KIT_Dataset(Dataset):
    def __init__(self, train_list):
        self.train_list = []

        for entry in train_list:
            x = torch.cat(entry[0]).float()
            y = torch.tensor(entry[1]).float()
            self.train_list.append((x,y))

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        return self.train_list[idx][0], self.train_list[idx][1]
    
from torch.utils.data import DataLoader
train_set = KIT_Dataset(new_train)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


from torch import nn
import torch.nn.functional as F


class ReshapeLayer(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[1])

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(2860, 64), # previously 2727
#             nn.BatchNorm1d(64),
#             ReshapeLayer(),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 8),
#             nn.BatchNorm1d(8),
#             ReshapeLayer(),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(8, 1),
#         )

#     def forward(self, x):
#         pred = self.linear_relu_stack(x)
#         return pred
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2860, 512)
        self.batchNorm1d_1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 32)
        self.batchNorm1d_2 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32, 1)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batchNorm1d_1(x)
        # x = x.squeeze()
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.batchNorm1d_2(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

from copy import deepcopy


model = NeuralNetwork().to(device)

learning_rate = 5e-4
epochs = 2000

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()

best_model = None
best_loss = 9999.999
best_epoch = 0

for e in range(epochs):
    total_loss = 0.0
    batches = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        loss = loss_fn(pred, y.unsqueeze(dim=1))
        total_loss += loss.item()
        batches += 1

        loss.backward()
        optimizer.step()
    total_loss /= batches
    print("Epoch", e+1, "of", epochs, "    loss:", total_loss)
    
    if total_loss < best_loss:
        best_model = deepcopy(model)
        best_loss = total_loss
        best_epoch = e + 1

print("\nBest epoch:", best_epoch, "best loss:", best_loss)

new_test = []

for idx, entry in enumerate(test):
    smiles = entry

    mol = Chem.MolFromSmiles(smiles)
    sentence = mol2alt_sentence(mol, radius=1) # I MIGHT need to convert to list before next line

    mol2vec_embedding = sentences2vec([sentence], mol2vec_model, unseen='UNK')
    reshaped_mol2vec_embedding = mol2vec_embedding.reshape(300)
    
    # try:
    #     maccs = torch.tensor(list(MACCSkeys.GenMACCSKeys(mol))).float()
    # except:
    #     print(smiles)
    #     continue

    # new_test.append( ([ maccs, wt[0], wt[1] ], (0, idx)) )
    # new_test.append( ([ maccs, ti670[0], ti670[1] ], (670, idx)) )
    # new_test.append( ([ maccs, vg560[0], vg560[1] ], (560, idx)) )

    new_test.append( ([ torch.tensor(reshaped_mol2vec_embedding).double(), wt[0], wt[1] ], (0, idx)) )
    new_test.append( ([ torch.tensor(reshaped_mol2vec_embedding).double(), ti670[0], ti670[1] ], (670, idx)) )
    new_test.append( ([ torch.tensor(reshaped_mol2vec_embedding).double(), vg560[0], vg560[1] ], (560, idx)) )


test_set = KIT_Dataset(new_test)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

{k: pred[:, 0] for idx, k in enumerate(benchmark.target_cols)}

import numpy as np

test_prob = {
    'CLASS_KIT' : np.empty(87),
    'CLASS_KIT_(T6701_mutant)' : np.empty(87),
    'CLASS_KIT_(V560G_mutant)' : np.empty(87)
}


with torch.no_grad():
    best_model.eval()
    for x, y in test_loader:
        x = x.to(device)
        pred = best_model(x)
        pred = nn.functional.sigmoid(pred).item()
        target = y[0,0].item()
        idx = y[0,1].int().item()

        if target == 0:
            test_prob['CLASS_KIT'][idx] = pred
        if target == 670:
            test_prob['CLASS_KIT_(T6701_mutant)'][idx] = pred
        if target == 560:
            test_prob['CLASS_KIT_(V560G_mutant)'][idx] = pred

test_pred = deepcopy(test_prob)
for key in test_pred.keys():
    for idx, entry in enumerate(test_pred[key]):
        test_pred[key][idx] = 1 if entry > 0.7 else 0

    print( sum(test_pred[key]))

results = benchmark.evaluate(y_pred=test_pred, y_prob=test_prob)
results.name = "KinaseSelectivityTeam1"
# results.description = "MACCS, ESM with single position and simple MLP"
results.description = "Mol2Vec, ESM with single position and simple MLP"
results.github_url = "https://github.com/william2343/ml4dd/edit/main/ml4dd_team1.py"
results.paper_url = "https://docs.google.com/document/d/1fQDLgrngs0v5iFbJ8CPW3c4G-y1WXogUn8nfpC1a968/edit?usp=sharing"
results.contributors = ["cynthia-habo", "whong", "hackefleisch", "natyyffo"]
results



results.upload_to_hub(owner='mldd-team1')
print("Upload?")
