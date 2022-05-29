import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("data/sp/sp_oall_32k.model")

def tokenize(raw_text):
    tokenized = sp.EncodeAsPieces(raw_text)
    return tokenized

prefs = ['train', 'valid', 'test']
expands = ['src', 'dst']
perchat_dir = 'data/perchat'

for p in prefs:
    for e in expands:
        lines = []
        with open(f'{perchat_dir}/{p}.{e}', mode='r') as f:
            lines = f.readlines()
            liens = [ line.strip('\n') for line in lines ]

            # Tokenize each text
            lines = [ tokenize(line) for line in lines ]

        with open(f'{perchat_dir}/{p}.{e}', mode='w') as f:
            f.write('\n'.join(lines))