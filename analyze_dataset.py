from dataset.bookcorpus import BookCorpusDataset
from dataset.tokenizer import Tokenizer

MAX_SEQ_LEN = 128

def get_sample(dataset):
    for batch in dataset:
        return batch[0, :]

tokenizer = Tokenizer()
train_dataset = BookCorpusDataset("train", tokenizer, MAX_SEQ_LEN, do_shuffle=False, allow_overlap=True)

val_dataset = BookCorpusDataset("val", tokenizer, MAX_SEQ_LEN, do_shuffle=False, allow_overlap=False)

test_dataset = BookCorpusDataset("test", tokenizer, MAX_SEQ_LEN, do_shuffle=False, allow_overlap=False)

print("Length: ", len(train_dataset))
print("Sample: ", tokenizer.decode(get_sample(train_dataset).tolist()))
print("Length: ", len(val_dataset))
print("Sample: ", tokenizer.decode(get_sample(val_dataset).tolist()))
print("Length: ", len(test_dataset))
print("Sample: ", tokenizer.decode(get_sample(test_dataset).tolist()))

# splits = BookCorpusDataset.get_splits(tokenizer, 128)