from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

# Initializing a TTT ttt-1b style configuration
# configuration = TTTConfig(**TTT_STANDARD_CONFIGS['1b']) is equivalent to the following
configuration = TTTConfig()
# print(configuration)

# Initializing a model from the ttt-1b style configuration
model = TTTForCausalLM(configuration)
model.eval()

# Accessing the model configuration
configuration = model.config

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Prefill
# text = "Greeting from TTT!" # Sequence = 7 tokens
text = "Greetings from TTT! What about going to the club to enjoy the basketball?"

input_ids = tokenizer(text, return_tensors="pt").input_ids
# logits = model(input_ids=input_ids)
# print(logits)

print("-----------------------------")

# Decoding
# out_ids = model.generate(input_ids=input_ids, max_length=50)
# out_ids = model.generate(input_ids=input_ids, max_length=9)
# out_ids = model.generate(input_ids=input_ids, max_length=10)
out_ids = model.generate(input_ids=input_ids, max_length=30)

out_str = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
print(out_str)

