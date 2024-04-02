import numpy as np
from tqdm import tqdm

def evaluate(dataset, model, **kwargs):
    # This code was adapted from Taylor Webb's code:
    # https://github.com/taylorwwebb/emergent_analogies_LLM/blob/main/digit_mat/eval_gpt_matprob.py

    all_prob = dataset.items
    all_prob_types = list(all_prob["all_problems"].item().keys())

    # Create data structure for storing results
    data = []

    # Loop over all problem indices
    N_prob = 40
    for prob_ind in tqdm(range(N_prob)):
        # Loop over all problem types
        for p in range(len(all_prob_types)):
            # Problem type
            prob_type = all_prob_types[p]
            print('Problem type: ' + prob_type + '...')
            perm_invariant = all_prob['all_problems'].item()[prob_type]['perm_invariant']
            prob_type_N_prob = all_prob['all_problems'].item()[prob_type]['prob'].shape[0]
            if prob_ind < prob_type_N_prob: # and len(all_gen_correct_pred[prob_type]) <= prob_ind:

                # Initialize results dictionary for this particular problem.
                res = {
                    "problem_type": prob_type,
                    "problem_ind": prob_ind,
                    "problem_id": str(prob_type) + "_" + str(prob_ind),
                    "perm_invariant": perm_invariant
                }

                # Problem
                prob = all_prob['all_problems'].item()[prob_type]['prob'][prob_ind]
                answer_choices = all_prob['all_problems'].item()[prob_type]['answer_choices'][prob_ind]
                correct_ind = all_prob['all_problems'].item()[prob_type]['correct_ind'][prob_ind]
                correct_answer = answer_choices[correct_ind]
                res["correct_answer"] = correct_answer

                # Generate prompt
                prompt = ''
                for r in range(3):
                    for c in range(3):
                        prompt += '['
                        if not (r == 2 and c == 2):
                            for i in range(len(prob[r][c])):
                                if prob[r][c][i] == -1:
                                    prompt += ' '
                                else:
                                    prompt += str(prob[r][c][i])
                                if i < len(prob[r][c]) - 1:
                                    prompt += ' '
                            prompt += ']'
                            if c < 2:
                                prompt += ' '
                            else:
                                prompt += '\n'

                # EVAL METHOD 1: Free text generation
                # Get response
                prediction = model.generate(prompt, **kwargs)[0]
                # Get log prob of generated output
                gen_logprob = model.get_logprob_of_continuation(
                    [prompt], 
                    [prediction], 
                    separator="",
                    reduction="sum"
                )[0]
                res["generated_response"] = prediction
                res["generated_response_sum_logprob"] = gen_logprob
                
                # Get prediction set
                pred_set = []
                invalid_char = False
                for c in prediction:
                    if c != ' ':
                        if c.isdigit():
                            pred_set.append(int(c))
                        elif c == ']':
                            # Stop after closing bracket
                            break
                        else:
                            invalid_char = True
                            break

                # Sort answer if problem type is permutation invariant
                if perm_invariant:
                    correct_answer = np.sort(correct_answer)
                    pred_set = np.sort(pred_set)
                # Determine whether prediction is correct
                correct_pred = False
                if not invalid_char and len(pred_set) == len(correct_answer):
                    if np.all(pred_set == correct_answer):
                        correct_pred = True

                res["generated_response_pred_set"] = pred_set
                res["generated_response_correct"] = correct_pred

                # EVAL METHOD 2: Forced-choice scoring
                continuations = []
                for a in range(8):
                    # Convert choice to string and remove ','
                    choice_str = np.array(str(answer_choices[a]))
                    choice_str = ''.join(list(choice_str[choice_str != ',']))
                    continuations.append(choice_str[1:]) # don't include opening bracket

                prefixes = [prompt for _ in continuations]
                all_choice_logprob = model.get_logprob_of_continuation(
                    prefixes, 
                    continuations, 
                    separator="",
                    reduction="sum"
                )
                res["mc_answer_logprobs"] = all_choice_logprob
                # Select answer
                model_choice = np.argmax(all_choice_logprob)
                res["mc_response_index"] = model_choice
                res["mc_response"] = continuations[model_choice]
                res["mc_response_sum_logprob"] = all_choice_logprob[model_choice]

                # Determine whether multiple choice selection is correct
                MC_correct = model_choice == correct_ind
                res["mc_response_correct"] = MC_correct
                data.append(res)

    return data