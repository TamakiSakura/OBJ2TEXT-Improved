
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<start>'
EOS_WORD = '<end>'

def set_constant(vocab):
    global UNK
    global BOS
    global EOS
    UNK = vocab(UNK_WORD)
    BOS = vocab(BOS_WORD)
    EOS = vocab(EOS_WORD)
