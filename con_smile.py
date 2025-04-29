from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import pandas
data = pandas.read_csv('smile.csv',header=None)
cano_list = []
i = 0
for row in data.values:
    print(i)
    smi = row[0]
    print(smi)
    mol = Chem.MolFromSmiles(smi)
    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    cano_list.append(canonical_smi)
    i += 1
df = pandas.DataFrame(cano_list)
df.to_csv("canon.csv")

