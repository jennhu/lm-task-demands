from tqdm import tqdm

def get_metalinguistic_prompt_paired(sentence1: str, sentence2: str) -> str:
    prompt = f"""
Which of the following two sentences is more grammatically correct in English? Respond with 1 or 2 as your answer.
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Answer:"""
    return prompt


def evaluate(dataset, model, condition="paired", **kwargs):
    # Initialize storage for results.
    data = []

    # Fetch items from dataset.
    stimuli = dataset.items
    meta_vars = list(stimuli)

    # Evaluate
    for _, example in tqdm(stimuli.iterrows(), total=dataset.n_items):
        # Initialize meta information for this item
        res = {v: example[v] for v in meta_vars}
        res["condition"] = condition

        # Get minimal pair.
        sentence_good = example["sentence_grammatical"] + "."
        sentence_bad= example["sentence_ungrammatical"] + "."

        ########################################################################
        # Direct method: minimal pair comparison
        ########################################################################
        
        sentences = [sentence_good, sentence_bad]
        sentence_probs = model.get_logprob_of_sequence(
            sentences,
            reduction="sum"
        )
        res["direct_sum_logprob_good_sentence"] = sentence_probs[0]
        res["direct_sum_logprob_bad_sentence"] = sentence_probs[1]
        res["direct_correct"] = sentence_probs[0] > sentence_probs[1]
        
        ########################################################################
        # Metalinguistic method
        ########################################################################

        # Set up prefixes and continuations.
        good_first_prompt = get_metalinguistic_prompt_paired(sentence_good, sentence_bad)
        bad_first_prompt = get_metalinguistic_prompt_paired(sentence_bad, sentence_good)

        prefixes = [
            good_first_prompt,
            good_first_prompt,
            bad_first_prompt,
            bad_first_prompt
        ]
        continuations = [
            "1",
            "2",
            "1",
            "2"
        ]
        # goodfirst/1, goodfirst/2, badfirst/1, badfirst/2
        G1, G2, B1, B2 = model.get_logprob_of_continuation(
            prefixes, 
            continuations,
            separator=" ",
            reduction="sum"
        )
        res["meta_sum_logprob_good_first_1"] = G1
        res["meta_sum_logprob_good_first_2"] = G2
        res["meta_sum_logprob_bad_first_1"] = B1
        res["meta_sum_logprob_bad_first_2"] = B2
        res["meta_1_correct"] = G1 > B1
        res["meta_2_correct"] = B2 > G2
        res["meta_good_correct"] = G1 > G2
        res["meta_bad_correct"] = B2 > B1

        # Strict and lenient item-level scores.
        # During analysis, we'll average over meta_good_correct and meta_bad_correct.
        res["meta_correct"] = (res["meta_good_correct"] and res["meta_bad_correct"])
        res["meta_correct_lenient"] = (res["meta_good_correct"] or res["meta_bad_correct"])

        # Store all the results from this item.
        data.append(res)

    return data