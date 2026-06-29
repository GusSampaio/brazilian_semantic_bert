import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def test_srl_model(model_repo_id, input_sentence):
    print(f"Downloading model and its tokenizer at: {model_repo_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    model = AutoModelForTokenClassification.from_pretrained(model_repo_id)
    model.eval()
    
    print("\nProcessing input sentence...")
    inputs = tokenizer(input_sentence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print("\nResults from Semantic Roles Classification:")
    print(f"{'Word token':<20} | {'Classified Semantic Role':<25}")
    print("-" * 50)
    
    current_word = ""
    current_tag = ""
    
    for token, pred_id in zip(tokens, predictions):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "<PRED>", "</PRED>"]:
            continue
            
        label = model.config.id2label[pred_id]
        
        
        if token.startswith("##"): # Leading with subword tokens 
            current_word += token.replace("##", "")
        else:
            if current_word:
                print(f"{current_word:<20} | {current_tag:<25}")
                
            current_word = token
            if token in ["<", "PRED", ">", "</"]: 
                current_word = token
            
            current_tag = label
            
    if current_word:
        print(f"{current_word:<20} | {current_tag:<25}")

if __name__ == "__main__":
    MODEL_REPO_ID = "GusSampaio/bert-base-portuguese-cased-srl"
    INPUT = "A Joana <PRED> caiu </PRED> da escada ontem."
    test_srl_model(MODEL_REPO_ID, INPUT)