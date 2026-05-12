def PBP_parser(filepath):
    sentences = []
    current_tokens = []
    
    current_rows = []
    current_meta = {}
    
    # Read and separate by sentences
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # fim da sentença
            # end of sentence
            if not line:
                if current_rows:
                    sentences.append({
                        "meta": current_meta,
                        "rows": current_rows
                    })
                    
                    current_rows = []
                    current_meta = {}
                continue
            
            # metadata
            if line.startswith("#"):

                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    current_meta[key.strip()] = value.strip()
                continue


            # ignore multiword tokens
            if "-" in line.split("\t")[0]:
                continue  # ignore intervals like 13-14
            
            parts = line.split("\t")
            
            if len(parts) < 10:
                continue

            current_rows.append(parts)
        
        # add the last sentence if the file doesn't end with a blank line
        if current_rows:
            sentences.append({
                "meta": current_meta,
                "rows": current_rows
            })

    flat_instances = []

    # Process each sentence
    for sent in sentences:
        rows = sent["rows"]
        meta = sent["meta"]
        tokens = [r[1] for r in rows]
        preds = {}
        args_by_pred = {}

        # Identify predicates
        for r in rows:
            idx, form, lemma, upos, _, feats, head, deprel, pred, role = r
            if pred != "_" and pred.strip():
                preds[int(idx)] = pred
                args_by_pred[int(idx)] = {}

        # Inference of implicit predicates via arguments
        for r in rows:
            role = r[9]

            if role == "_" or not role.strip():
                continue
        
            for part in role.split("|"):
                if ":" not in part:
                    continue
        
                _, pred_id = part.split(":")
        
                if not pred_id.isdigit():
                    continue
        
                pred_id = int(pred_id)
        
                # already exists explicitly
                if pred_id in preds:
                    continue
                
                pred_row = rows[pred_id - 1]
                pred_lemma = pred_row[2]
                preds[pred_id] = pred_lemma
                args_by_pred[pred_id] = {}

        # Collect roles linked to each predicate
        for r in rows:
            idx, form, lemma, upos, _, feats, head, deprel, pred, role = r
            if role != "_" and role.strip():
                for part in role.split("|"):
                    if ":" not in part:
                        continue
                    role_label, pred_id = part.split(":")
                    if pred_id.isdigit() and int(pred_id) in args_by_pred:
                        args_by_pred[int(pred_id)].setdefault(role_label, []).append({
                            "token_id": int(idx),
                            "token": form
                        })

        # flat structure for training
        for pred_idx, pred_id in preds.items():
            labels = []
            for r in rows:
                form = r[1]
                role_str = r[9]
                label = "O"
                if str(pred_idx) == r[0]:
                    label = "PRED"
                
                elif role_str != "_":    
                    for part in role_str.split("|"):

                        if ":" not in part:
                            continue

                        role_name, linked_pred = part.split(":")

                        if int(linked_pred) == pred_idx:
                            label = role_name.upper()
                            break
                            
                labels.append(label)
                
            flat_instances.append({
                "sentence_id": meta.get("sent_id"),
                "text": meta.get("text"),
                "predicate_index": pred_idx-1, # adjust to 0-based index
                "predicate_token": tokens[pred_idx - 1],
                "predicate_lemma": preds[pred_idx].split(".")[0],
                "predicate_frame": pred_id,
                "tokens": tokens,
                "labels": labels
            })

    return flat_instances