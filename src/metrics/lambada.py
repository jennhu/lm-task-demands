from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn.functional as F
import torch

def get_metalinguistic_prompt(prefix: str) -> str:
    prompt = f"What word is most likely to come next in the following text?\nText: {prefix}\nPrediction:"
    return prompt

def logits_to_dist(logits):
    return F.softmax(logits, dim=-1)

def evaluate(dataset, model, k=100, **kwargs):
    # Initialize storage for results.
    data = []

    # Fetch items from Lambada dataset.
    stimuli = dataset.items

    # Evaluate
    for i, example in tqdm(enumerate(stimuli), total=dataset.n_items):
        text = example["text"]

        # Get final word to be predicted (by splitting on whitespace).
        # NOTE: there's some debate about what the "true" Lambada task is:
        # https://github.com/EleutherAI/lm-evaluation-harness/issues/350
        splits = text.split(" ")
        prefix = " ".join(splits[:-1])
        final_word = splits[-1]

        # Initialize meta information for this item
        res = {"item_id": i, "prefix": prefix, "final_word": final_word}

        # Make prompt.
        prompt = get_metalinguistic_prompt(prefix)

        ########################################################################
        # Get logprob of final word
        ########################################################################
        # Set up prefixes and continuations for batched inference.
        prefixes = [prefix, prompt]
        continuations = [final_word for _ in prefixes]
        direct_logprob, meta_logprob = model.get_logprob_of_continuation(
            prefixes, 
            continuations,
            separator=" ",
            reduction="sum"
        )
        res["direct_logprob"] = direct_logprob
        res["meta_logprob"] = meta_logprob

        ########################################################################
        # Compare distributions
        ########################################################################
        direct_logits = model.predict(prefix)[0, -1, :] # 1-dim: vocab size
        meta_logits = model.predict(prompt)[0, -1, :] # 1-dim: vocab size

        # 1. KL-divergence between next-word distributions.
        direct_logprobs = F.log_softmax(direct_logits, dim=0)
        meta_logprobs = F.log_softmax(meta_logits, dim=0)
        kl = F.kl_div(
            meta_logprobs, 
            direct_logprobs, # treated as the "true" target
            log_target=True, 
            reduction="batchmean"
        ).item()
        res["kl_divergence"] = kl

        # 2. Rank order correlation (all tokens)
        r, p = spearmanr(direct_logits.cpu().numpy(), meta_logits.cpu().numpy())
        res["spearman_r_all"] = r
        res["spearman_p_all"] = p

        # 3. Rank order correlation (top 100 tokens under direct distribution)
        _, topk_inds = torch.topk(direct_logits, k)
        sorted_topk_inds = torch.stack(sorted(topk_inds))
        direct_top = direct_logits[sorted_topk_inds]
        meta_top = meta_logits[sorted_topk_inds]
        r, p = spearmanr(direct_top.cpu().numpy(), meta_top.cpu().numpy())
        res[f"spearman_r_top{k}"] = r
        res[f"spearman_p_top{k}"] = p

        # Store all the results from this item.
        data.append(res)

    return data