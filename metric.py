def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start
def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def f1_score(y_true, y_pred, average='macro',suffix=False):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))
    if average == 'micro':
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
        r = 100 * nb_correct / nb_true if nb_true > 0 else 0
        score = 2 * p * r / (p + r) if p + r > 0 else 0
        return p, r, score
    else:
        f_scores = {}
        p_scores = {}
        r_scpres = {}
        labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        for label in labels:
            true_entities4a_label = set()
            pred_entities4a_label = set()
            for te in true_entities:
                if te[0] == label:
                    true_entities4a_label.add(te)
            for pe in pred_entities:
                if pe[0] == label:
                    pred_entities4a_label.add(pe)
            nb_correct = len(true_entities4a_label & pred_entities4a_label)
            nb_pred = len(pred_entities4a_label)
            nb_true = len(true_entities4a_label)

            p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
            r = 100 * nb_correct / nb_true if nb_true > 0 else 0
            score = 2 * p * r / (p + r) if p + r > 0 else 0
            f_scores[label] = round(score,2)
            p_scores[label] = round(p,2)
            r_scpres[label] = round(r,2)
        p = round(sum(p_scores.values()) / len(p_scores),2)
        r = round(sum(r_scpres.values()) / len(r_scpres),2)
        f1 = round(sum(f_scores.values()) / len(f_scores),2)
        return f_scores, p_scores, r_scpres, p, r, f1

from sklearn.metrics import confusion_matrix
from texttable import Texttable
def ner_confusion_matrix(golden, pred, suffix=False):
    labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene', 'O']
    true_entities = set(get_entities(golden, suffix))
    pred_entities = set(get_entities(pred, suffix))
    true_span2type = {(i, j): k for k, i, j in true_entities}
    pred_span2type = {(i, j): k for k, i, j in pred_entities}

    add_span = set()
    pred_label = []
    true_label = []
    # 根据span判断pred和golden的对应预测标签，如果span在golden中，但是没在pred中，那么认为pred为O
    for span, label in true_span2type.items():
        true_label.append(label)
        pred_label.append(pred_span2type.get(span, 'O'))
        add_span.add(span)
    # span在pred中没在gold中，那么认为gold为O
    for span, label in pred_span2type.items():
        if span in add_span:
            continue
        pred_label.append(label)
        true_label.append(true_span2type.get(span, 'O'))
    cf_matrix =  confusion_matrix(true_label, pred_label, labels=labels)
    table = Texttable()
    table.add_row([" "] + [i[:4] for i in labels])
    table.set_max_width(2000)
    for idx, r in enumerate(cf_matrix):
        table.add_row([labels[idx][:4]] + [str(i) for i in cf_matrix[idx]])
    return table.draw()
