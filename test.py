from huggingface_hub import login
from tokenizers import Tokenizer

login()

tokenizer = Tokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=77)
tokenizer.enable_truncation(max_length=77)


print(tokenizer.get_vocab_size())


encode = tokenizer.encode_batch_fast(["this is test for clip engram tokenizer"])

print(encode[0].ids)
