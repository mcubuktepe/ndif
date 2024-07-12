
import nnsight
nnsight.CONFIG.set_default_api_key('0Bb6oUQxj2TuPtlrTkny')


model = nnsight.LanguageModel("meta-llama/Meta-Llama-3-70B")


with model.trace("ayy", remote=True):
    
    output = model.output.save()
model = nnsight.LanguageModel("meta-llama/Meta-Llama-3-8B")


with model.trace("ayy", remote=True):
    
    output = model.output.save()
    
model = nnsight.LanguageModel("EleutherAI/gpt-j-6b")


with model.trace("ayy", remote=True):
    
    output = model.output.save()
    
# model = nnsight.LanguageModel("openai-community/gpt2")


# with model.trace("ayy", remote=True):
    
#     output = model.output.save()
    
