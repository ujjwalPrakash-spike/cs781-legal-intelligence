#!/usr/bin/env python3
"""04_phase2_offset_alignment_demo.py
Demonstrates byte-pair token-to-character offset alignment for Legal NER,
showing why naive substring searching fails and how offset mapping prevents 0.0 scores.
"""

def extract_entities_from_offsets(tokens, offset_mapping, predicted_bio_tags, raw_text):
    """Reconstructs exact character start and end spans from token-level BIO tags."""
    entities = []
    current_entity = None
    
    for (start_char, end_char), tag in zip(offset_mapping, predicted_bio_tags):
        if start_char == end_char:  # Special token (BOS, EOS, PAD)
            continue
            
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]
            current_entity = {
                'entity': entity_type,
                'start': start_char,
                'end': end_char
            }
        elif tag.startswith('I-') and current_entity and current_entity['entity'] == tag[2:]:
            current_entity['end'] = end_char  # Extend span to include this token
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
                
    if current_entity:
        entities.append(current_entity)
        
    # Extract matching substring directly from raw text using character offsets
    for ent in entities:
        span_text = raw_text[ent['start']:ent['end']]
        # Strip leading/trailing whitespaces and adjust character offsets accordingly
        l_stripped = len(span_text) - len(span_text.lstrip())
        r_stripped = len(span_text) - len(span_text.rstrip())
        ent['start'] += l_stripped
        ent['end'] -= r_stripped
        ent['text'] = raw_text[ent['start']:ent['end']]
        
    return entities

if __name__ == "__main__":
    print("=== Phase 2: Byte-Level Character Offset Recovery Demo ===")
    
    # Authentic Indian legal sentence with multi-word entities and numbers
    raw_document = "The Supreme Court of India heard the petition under Section 302 of the IPC."
    
    # Simulated subword tokens and their exact character start/end coordinates
    # "The" -> (0, 3), " Supreme" -> (3, 11), " Court" -> (11, 17), " of" -> (17, 20), " India" -> (20, 26)
    tokens = ["The", " Supreme", " Court", " of", " India", " heard", " the", " petition", " under", " Section", " 302", " of", " the", " IPC", "."]
    offsets = [
        (0, 3), (3, 11), (11, 17), (17, 20), (20, 26),
        (26, 32), (32, 36), (36, 45), (45, 51),
        (51, 59), (59, 63), (63, 66), (66, 70), (70, 74), (74, 75)
    ]
    bio_tags = [
        "O", "B-COURT", "I-COURT", "I-COURT", "I-COURT",
        "O", "O", "O", "O",
        "B-PROVISION", "I-PROVISION", "O", "O", "B-STATUTE", "O"
    ]
    
    extracted = extract_entities_from_offsets(tokens, offsets, bio_tags, raw_document)
    
    print(f"Raw Document: '{raw_document}'\n")
    print("Extracted Spans:")
    for ent in extracted:
        print(f"  [{ent['start']:2d}:{ent['end']:2d}] {ent['entity']:<12} -> '{ent['text']}'")
        
    assert extracted[0]['text'] == "Supreme Court of India"
    assert extracted[1]['text'] == "Section 302"
    assert extracted[2]['text'] == "IPC"
    print("\nSUCCESS: Exact character boundaries matched source string perfectly.")
