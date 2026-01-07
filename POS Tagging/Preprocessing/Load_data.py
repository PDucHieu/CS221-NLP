from conllu import parse_incr

def load_conllu_data(path):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for sent in parse_incr(f):
            sentence = []
            for token in sent:
                if isinstance(token["id"], int):
                    if token["form"] and token["upostag"]:
                        sentence.append((token["form"], token["upostag"]))
            if sentence:
                sentences.append(sentence)
    return sentences
