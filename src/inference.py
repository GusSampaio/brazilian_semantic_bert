import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

class SRLPredictor:
    def __init__(self, model_path="artifacts/srl_model_final"):
        print(f"A carregar o modelo de '{model_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval() # Coloca o modelo em modo de inferência (desliga o dropout, etc.)
        self.id2label = self.model.config.id2label

    def _safe_get_label(self, idx):
        """Garante que a label é encontrada, quer o ID seja int ou string."""
        if int(idx) in self.id2label:
            return self.id2label[int(idx)]
        elif str(idx) in self.id2label:
            return self.id2label[str(idx)]
        return "O"

    def predict(self, text, predicate_index):
        """
        Recebe uma frase e o índice (0-based) da palavra que é o verbo/predicado.
        Retorna uma string formatada com os papéis semânticos.
        """
        # Para um caso real de produção, poderias usar o spaCy para tokenizar.
        # Aqui usamos o split() básico para simplificar.
        words = text.split()

        if predicate_index >= len(words):
            return "Erro: O índice do predicado é maior do que o número de palavras na frase."

        # 1. Inserir os Special Tokens (como faz o instance_builder)
        new_words = []
        for i, word in enumerate(words):
            if i == predicate_index:
                new_words.append("<PRED>")
                new_words.append(word)
                new_words.append("</PRED>")
            else:
                new_words.append(word)

        # 2. Tokenizar para o BERT
        encoding = self.tokenizer(
            new_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        # 3. Fazer o Forward Pass (Inferência)
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

        # 4. Alinhar as predições de volta para as palavras originais
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_id = None

        for i, word_id in enumerate(word_ids):
            # Ignora [CLS], [SEP] e paddings
            if word_id is None:
                continue
            
            # Só olhamos para a primeira subword de cada palavra
            if word_id != previous_word_id:
                actual_word = new_words[word_id]
                
                # Ignoramos a predição para os special tokens em si
                if actual_word not in ["<PRED>", "</PRED>"]:
                    pred_label = self._safe_get_label(predictions[i])
                    aligned_labels.append((actual_word, pred_label))
                    
            previous_word_id = word_id

        # 5. Formatar a saída (junta palavras consecutivas com a mesma label)
        return self._format_output(aligned_labels)

    def _format_output(self, aligned_labels):
        result = []
        current_label = None
        current_words = []

        for word, label in aligned_labels:
            if label == "O":
                if current_label is not None:
                    result.append(f"[{current_label}: {' '.join(current_words)}]")
                    current_label = None
                    current_words = []
                result.append(word)
            else:
                if label == current_label:
                    current_words.append(word)
                else:
                    if current_label is not None:
                        result.append(f"[{current_label}: {' '.join(current_words)}]")
                    current_label = label
                    current_words = [word]

        if current_label is not None:
            result.append(f"[{current_label}: {' '.join(current_words)}]")

        return " ".join(result)

if __name__ == "__main__":
    predictor = SRLPredictor()

    print("\n" + "="*50)
    print("Teste de Inferência de SRL")
    print("="*50)

    frase1 = "A Joana atirou a bola para o cão rapidamente."
    idx_verbo1 = 2 # 0:A, 1:Joana, 2:atirou
    print(f"\nFrase: {frase1}")
    print(f"Verbo: {frase1.split()[idx_verbo1]}")
    print(f"Resultado: {predictor.predict(frase1, idx_verbo1)}")

    # Exemplo 2
    frase2 = "Ontem, o ministro anunciou novas medidas econômicas na capital."
    idx_verbo2 = 4 # 0:Ontem, 1:,, 2:o, 3:ministro, 4:anunciou
    print(f"\nFrase: {frase2}")
    print(f"Verbo: {frase2.split()[idx_verbo2]}")
    print(f"Resultado: {predictor.predict(frase2, idx_verbo2)}")

    print("\nExperimente!")
    while True:
        try:
            texto = input("\nEscreve uma frase (ou Ctrl+C para sair): ")
            palavras = texto.split()
            for i, p in enumerate(palavras):
                print(f"[{i}] {p}")
            
            idx = int(input("Qual é o índice do verbo? "))
            print(f"\nPredição: {predictor.predict(texto, idx)}")
        except KeyboardInterrupt:
            print("\nA sair...")
            break
        except Exception as e:
            print(f"Erro: {e}")