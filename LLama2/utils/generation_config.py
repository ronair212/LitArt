import sys
sys.path.insert(1,'/home/nair.ro/test/LitArt/falcon')


def get_generation_config(model):
    model.generation_config.max_new_tokens = 200
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.7
    model.generation_config.num_return_sequences = 1
    model.generation_config.pad_token_id = model.tokenizer.eos_token_id
    model.generation_config.eos_token_id = model.tokenizer.eos_token_id
    return model.generation_config
