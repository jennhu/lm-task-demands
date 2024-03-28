from tqdm import tqdm
import numpy as np
import re

def clean_response(response):
    # Keep everything before newline character or period
    if "\n" in response:
        prediction = response.split("\n")[0]
        if "." in response:
            prediction = prediction.split(".")[0] + "."
    elif "." in response:
        prediction = response.split(".")[0] + "."
    else:
        prediction = response
    return prediction.strip()

def make_prompt(question):
    return f"Question: {question}\nAnswer:"

def format_as_money(num):
    if isinstance(num, str):
        if num.startswith("$"):
            num = float(num[1:])
        else:
            num = float(num)
    return '${:,.2f}'.format(num)

def check_if_response_matches_answer(response, answer, condition="crt1"):
    if condition == "crt1":
        # Get all substrings of response that are formatted as money string.
        response_money_strs = re.findall(r"\$\d+(?:\.\d+)?", response)
        
        # If any of these is equal to the answer, return true.
        if any([answer == m for m in response_money_strs]):
            return True
        # Check if stripping off .00 makes a difference.
        elif answer.endswith(".00") and any([answer.replace(".00", "") == m for m in response_money_strs]):
            return True
        else:
            return False
    else:
        return answer in response

def evaluate(dataset, model, condition="crt1", **kwargs):
    # Initialize storage for results.
    data = []

    # Fetch items from CRT dataset (Hagendorff et al. 2023).
    stimuli = dataset.items
    stimuli = stimuli[stimuli.condition==condition]

    # Evaluate
    for _, example in tqdm(stimuli.iterrows(), total=len(stimuli)):
        question = example["question"]

        # Initialize meta information for this item
        meta_vars = ["condition", "item_id", "question"]
        answer_vars = ["correct", "intuitive"]
        if condition.startswith("crt1"):
            # Add distractor items
            answer_vars += ["total_cost", "more"]
            # Add alternatives (with .00 removed)
            answer_vars += ["correct_alt", "intuitive_alt", "total_cost_alt", "more_alt"]
        elif condition.startswith("crt3"):
            answer_vars += ["t"]

        # Construct text of answer options for forced-choice evaluation.
        answer_options = {}
        for v in answer_vars:
            if condition.startswith("crt1"):
                if v.endswith("_alt"):
                    answer_text = format_as_money(example[v.replace("_alt", "")])
                    if answer_text.endswith(".00"):
                        answer_text = answer_text[:-3]
                else:
                    answer_text = format_as_money(example[v])
            else:
                answer_text = example[v]
            answer_options[v] = answer_text

        res = {v: example[v] for v in meta_vars}
        res.update(answer_options)

        prompt = make_prompt(question)

        ########################################################################
        # EVAL METHOD 1: FREE GENERATION
        ########################################################################

        # Get response.
        prediction = model.generate(prompt, **kwargs)[0]
        res["generated_response_raw"] = prediction
        # Clean the generated response.
        clean_prediction = clean_response(prediction)
        try:
            clean_prediction = format_as_money(clean_prediction)
        except:
            pass
        res["generated_response"] = clean_prediction
        # NAIVE LABEL: do some basic pattern-matching to guess what answer
        # option the generated response corresponds to. This is NOT foolproof,
        # so we'll do manual annotation later.
        response_label = None
        for v, answer_option in answer_options.items():
            match = check_if_response_matches_answer(
                clean_prediction, answer_option, condition=condition
            )
            if match:
                response_label = v
        res["generated_response_naive_label"] = response_label

        ########################################################################
        # EVAL METHOD 2: FORCED CHOICE
        ########################################################################
        continuations = [answer_options[v] for v in answer_vars]
        prefixes = [prompt for _ in continuations]
        choice_logprobs = model.get_logprob_of_continuation(
            prefixes, 
            continuations, 
            separator=" ",
            reduction="mean"
        )
        res["mc_answer_logprobs"] = choice_logprobs
        res["mc_answer_labels"] = answer_vars
        res["mc_correct_logprob"] = choice_logprobs[0]
        res["mc_intuitive_logprob"] = choice_logprobs[1]

        # Select answer
        model_choice = np.argmax(choice_logprobs)
        res["mc_response_index"] = model_choice
        res["mc_response_label"] = answer_vars[model_choice]
        res["mc_response"] = continuations[model_choice]
        res["mc_response_mean_logprob"] = choice_logprobs[model_choice]

        # Determine whether multiple choice selection is correct
        MC_correct = model_choice == 0 # because "correct" is always ordered first
        res["mc_response_correct"] = MC_correct

        if condition.startswith("crt1"):
            res["mc_correct_alt_logprob"] = choice_logprobs[4]
            res["mc_intuitive_alt_logprob"] = choice_logprobs[5]
            res["mc_response_correct_alt"] = model_choice == 4

        # Store all the results from this item.
        data.append(res)

    return data