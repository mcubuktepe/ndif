
import nnsight

model = nnsight.LanguageModel("meta-llama/Meta-Llama-3-8B")

nnsight.CONFIG.set_default_api_key('0Bb6oUQxj2TuPtlrTkny')

with model.trace("ayy", remote=True):
    
    output = model.output.save()
    
