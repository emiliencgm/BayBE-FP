def cache_pretrained_model():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import os
    os.makedirs("./from_pretrained", exist_ok=True)
    
    # chemberta-small
    AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir="./from_pretrained")
    AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", cache_dir="./from_pretrained")
    # chemberta-large
    AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-100M-MLM", cache_dir="./from_pretrained")
    AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-100M-MLM", cache_dir="./from_pretrained")
    
    # LLM
    from base.llm_utils import cache__get_model_and_tokenizer
    cache__get_model_and_tokenizer("t5-base")
    cache__get_model_and_tokenizer("WhereIsAI/UAE-Large-V1")
    cache__get_model_and_tokenizer("GT4SD/multitask-text-and-chemistry-t5-base-augm")

if __name__ == "__main__":
    cache_pretrained_model()