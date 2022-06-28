import sentencepiece as spm
import sys

perchat_src = sys.argv[1] #'data/perchat/raw'
perchat_dst = sys.argv[2] #'data/perchat/spaced'

sp = spm.SentencePieceProcessor()
sp.Load("data/sp/sp_oall_32k.model")

def tokenize(raw_text):
    tokenized = sp.EncodeAsPieces(raw_text)
    return ' '.join(tokenized)

prefs = ['train', 'valid', 'test']
expands = ['src', 'dst']

for p in prefs:
    for e in expands:
        lines = []
        with open(f'{perchat_src}/{p}.{e}', mode='r') as f:
            lines = f.readlines()
            lines = [ line.strip('\n') for line in lines ]

            # Tokenize each text
            lines = [ tokenize(line) for line in lines ]
        
        with open(f'{perchat_dst}/{p}.{e}', mode='w') as f:
            f.write('\n'.join(lines))
        
