import pandas as pd

from metrics import lambada, digit_mat, crt, dgl, blimp
from stimuli import TASK_TO_DATASET

EVAL_FNS = {
    "crt": crt.evaluate,
    "lambada": lambada.evaluate,
    "digit_mat": digit_mat.evaluate,
    "dgl": dgl.evaluate,
    "blimp": blimp.evaluate
}

def save_data_csv(result, model, outpath):
    df = pd.DataFrame(result)
    df["model"] = model.model_name
    df["revision"] = model.revision
    df["instruct"] = model.instruct
    df.to_csv(outpath, index=False)
    print(f"Saved results to {outpath}")

def evaluate(model, outpath, task, **kwargs):
    # Get eval function and dataset.
    eval_fn = EVAL_FNS[task]
    dataset = TASK_TO_DATASET[task](task)

    # Run evaluation function.
    result = eval_fn(dataset, model, **kwargs)

    # Save data.
    save_data_csv(result, model, outpath)