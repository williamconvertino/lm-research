from dataset.bookcorpus import BookCorpusDataset
from dataset.tokenizer import Tokenizer

tokenizer = Tokenizer()
splits = BookCorpusDataset.get_splits(tokenizer, 128)