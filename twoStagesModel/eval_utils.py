
from data_process import *
import os, json
from eval_zb import eval_acc
from pathlib import Path

@dataclass
class Prediction:
    """
    用来保存可能的预测结果, feature-result-prediction是一一对应的关系
    """
    feature_index: int
    start_index: int  # 预测的对应Sequence中的开始位置
    end_index: int  # 预测的对应Sequence中的结束位置
    start_logit: float  # 预测的开始得分
    end_logit: float  # 预测的截止得分
    text: str = None  # 对应的在原文中的文本,一开始为None,后面被计算
    orig_start_index: int = None
    orig_end_index: int = None
    final_score: float = None


def compute_predictions(
        args,
        all_examples: List[cvExample],
        all_features,
        all_results,
        global_step,
        n_best_size=30,
        max_answer_length=30,
        do_lower_case=False,
        output_path="./output/valid-result.jsonl",
):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.id].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result[0]] = result

    all_reses = []
    all_nbest_predict = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example.id]
        predictions = []

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.pid]

            start_indexes = _get_best_indexes(result[1], n_best_size)
            end_indexes = _get_best_indexes(result[2], n_best_size)

            if start_indexes[0] == 0 or end_indexes[0] == 0:
                continue

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    predictions.append(
                        Prediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result[1][start_index],
                            end_logit=result[2][end_index],
                        )
                    )

        if len(predictions) == 0:
            res = {"url": example.url, "key_points": [{"begin": 0, "outline": ""}]}
            all_reses.append(res)
            all_nbest_predict[example.url] = ['']
        else:

            predictions = sorted(predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
            seen_predictions = {}
            filtered_predictions = []
            span_covered = [0 for i in range(len(example.doc_tokens))]

            for prediction in predictions:
                if len(filtered_predictions) >= 30:
                    break

                feature = features[prediction.feature_index]
                tok_tokens = feature.tokens[prediction.start_index: (prediction.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[prediction.start_index]
                orig_doc_end = feature.token_to_orig_map[prediction.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                prediction.orig_start_index = orig_doc_start
                prediction.orig_end_index = orig_doc_end

                if final_text in seen_predictions:
                    continue

                if 1 in span_covered[orig_doc_start: (orig_doc_end + 1)]:
                    continue
                span_covered[orig_doc_start: (orig_doc_end + 1)] = [1 for i in range(orig_doc_start,
                                                                                     orig_doc_end + 1)]
                seen_predictions[final_text] = True
                prediction.text = final_text
                filtered_predictions.append(prediction)

            predictions = filtered_predictions

            assert len(predictions) > 0
            score_normalization(predictions)

            best_non_null_entry = None
            for p in predictions:
                if best_non_null_entry is None and p.text != "":
                    best_non_null_entry = p
                    break

            if best_non_null_entry == None:
                res = {"url": example.url, "key_points": [{"begin": 0, "outline": ""}]}
                all_reses.append(res)
                all_nbest_predict[example.url] = ['']
                continue

            key_points = []
            max_score = best_non_null_entry.final_score
            span_covered = [0 for i in range(len(example.doc_tokens))]
            for p in predictions:
                # if 1 not in span_covered[p.orig_start_index: (p.orig_end_index + 1)]:
                if p.final_score > (max_score * args.multi_span_threshold) \
                        and 1 not in span_covered[p.orig_start_index: (p.orig_end_index + 1)]:

                    kp = {}
                    bn = get_begin_num(example.doc_tokens, p.orig_start_index)

                    kp["begin"] = bn
                    # kp["outline"] = get_new_text(p.text)
                    kp["outline"] = p.text
                    key_points.append(kp)

                    span_covered[p.orig_start_index: (p.orig_end_index + 1)] = [1 for i in range(p.orig_start_index,
                                                                                                 p.orig_end_index + 1)]
            # all_predictions.append({"url": example.url, "key_points": key_points})
            key_points = sorted(key_points, key=lambda x: x["begin"], reverse=False)
            res = {"url": example.url, "key_points": key_points}
            all_reses.append(res)

            current_nbest = []
            for (i, prediction) in enumerate(predictions):
                output = collections.OrderedDict()
                output["text"] = prediction.text
                output["start_logit"] = prediction.start_logit
                output["end_logit"] = prediction.end_logit
                output["final_score"] = prediction.final_score
                current_nbest.append(output)
            all_nbest_predict[example.url] = current_nbest

    # 删除上次保存的预测文件
    my_file = Path(output_path)
    if my_file.exists():
        os.remove(my_file)

    for res in all_reses:
        with open(output_path, "a+") as outFile:
            outFile.write(json.dumps(res, ensure_ascii=False) + '\n')
    with open('./output/valid-nbest.jsonl', "w", encoding="utf-8") as writer:
        writer.write(json.dumps(all_nbest_predict, indent=4, ensure_ascii=False))

    f1 = eval_acc("./data/new-valid.jsonl", output_path)
    return f1


def compute_predictions_in_dev(
        args,
        all_examples: List[cvExample],
        all_features,
        all_results,
        global_step,
        index,
        n_best_size=30,
        max_answer_length=30,
        do_lower_case=False,
):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.id].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result[0]] = result

    paths = './output/dev-result-{}.jsonl'.format(index)
    my_file = Path(paths)
    if my_file.exists():
        os.remove(my_file)
    nbest_path = './output/dev-nbest-{}.jsonl'.format(index)
    my_file = Path(nbest_path)
    if my_file.exists():
        os.remove(my_file)

    all_predictions = []
    all_nbest_predict = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example.id]
        predictions = []

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.pid]

            start_indexes = _get_best_indexes(result[1], n_best_size)
            end_indexes = _get_best_indexes(result[2], n_best_size)

            if start_indexes[0] == 0 or end_indexes[0] == 0:
                continue

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    predictions.append(
                        Prediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result[1][start_index],
                            end_logit=result[2][end_index],
                        )
                    )

        if len(predictions) == 0:
            res = {"url": example.url, "key_points": [{"begin": 0, "outline": ""}]}
            with open(paths, "a+") as outFile:
                outFile.write(json.dumps(res, ensure_ascii=False) + '\n')
            all_nbest_predict[example.url] = ['']
        else:

            predictions = sorted(predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
            seen_predictions = {}
            filtered_predictions = []
            span_covered = [0 for i in range(len(example.doc_tokens))]

            for prediction in predictions:
                if len(filtered_predictions) >= 30:
                    break

                feature = features[prediction.feature_index]
                tok_tokens = feature.tokens[prediction.start_index: (prediction.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[prediction.start_index]
                orig_doc_end = feature.token_to_orig_map[prediction.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                prediction.orig_start_index = orig_doc_start
                prediction.orig_end_index = orig_doc_end

                if final_text in seen_predictions:
                    continue

                if 1 in span_covered[orig_doc_start: (orig_doc_end + 1)]:
                    continue
                span_covered[orig_doc_start: (orig_doc_end + 1)] = [1 for i in range(orig_doc_start,
                                                                                     orig_doc_end + 1)]
                seen_predictions[final_text] = True
                prediction.text = final_text
                filtered_predictions.append(prediction)

            predictions = filtered_predictions

            assert len(predictions) > 0
            score_normalization(predictions)

            best_non_null_entry = None
            for p in predictions:
                if best_non_null_entry is None and p.text != "":
                    best_non_null_entry = p
                    break

            if best_non_null_entry == None:
                res = {"url": example.url, "key_points": [{"begin": 0, "outline": ""}]}
                with open(paths, "a+") as outFile:
                    outFile.write(json.dumps(res, ensure_ascii=False) + '\n')
                all_nbest_predict[example.url] = ['']
                continue

            key_points = []
            max_score = best_non_null_entry.final_score
            span_covered = [0 for i in range(len(example.doc_tokens))]
            for p in predictions:
                # if 1 not in span_covered[p.orig_start_index: (p.orig_end_index + 1)]:
                if p.final_score > (max_score * args.multi_span_threshold) \
                        and 1 not in span_covered[p.orig_start_index: (p.orig_end_index + 1)]:

                    kp = {}
                    bn = get_begin_num(example.doc_tokens, p.orig_start_index)

                    kp["begin"] = bn
                    # kp["outline"] = get_new_text(p.text)
                    kp["outline"] = p.text
                    key_points.append(kp)

                    span_covered[p.orig_start_index: (p.orig_end_index + 1)] = [1 for i in range(p.orig_start_index,
                                                                                                 p.orig_end_index + 1)]
            # all_predictions.append({"url": example.url, "key_points": key_points})
            key_points = sorted(key_points, key=lambda x: x["begin"], reverse=False)
            res = {"url": example.url, "key_points": key_points}
            with open(paths, "a+") as outFile:
                outFile.write(json.dumps(res, ensure_ascii=False) + '\n')
            current_nbest = []
            for (i, prediction) in enumerate(predictions):
                output = collections.OrderedDict()
                output["text"] = prediction.text
                output["start_logit"] = prediction.start_logit
                output["end_logit"] = prediction.end_logit
                output["final_score"] = prediction.final_score
                current_nbest.append(output)
            all_nbest_predict[example.url] = current_nbest


    with open(nbest_path, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(all_nbest_predict, indent=4, ensure_ascii=False))

    return 0


def get_new_text(text):

    while '[' in text and ']' in text:
        a = text.index('[')
        b = text.index(']')
        if a<b and text[a+1: b].isdigit():
            dt = text[a: b + 1]
            text = text.replace(dt, '')
        else:
            break
    return text


def get_begin_num(doc_tokens, index):
    res = 0
    if doc_tokens[index] == '[':
        if doc_tokens[index+1].isdigit():
            res = int(doc_tokens[index+1])
            return res
    if doc_tokens[index].isdigit() and index+1 < len(doc_tokens):
        if doc_tokens[index+1] == ']' and doc_tokens[index-1] == '[':
            res = int(doc_tokens[index])
            return res
    for i in range(index-1, 0, -1):
        if doc_tokens[i+1] == ']' and doc_tokens[i-1] == '[' and doc_tokens[i].isdigit():
            res = int(doc_tokens[i])
            return res
    return res


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logging.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logging.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


def score_normalization(predictions: List[Prediction]):
    scores = [p.start_logit + p.end_logit for p in predictions]
    max_score = max(scores)
    min_score = min(scores)
    for p in predictions:
        if (max_score - min_score) == 0:
            p.final_score = 0
            continue
        p.final_score = 1.0 * ((p.start_logit + p.end_logit) - min_score) / (max_score - min_score)