import configargparse
from pathlib import Path

from models import LM
import evaluate


def parse_args():
    """Parses command-line arguments for run_experiment.py."""
    parser = configargparse.ArgumentParser(description="Run experiment on LMs.")
    # File-related parameters
    parser.add("-c", "--config", is_config_file=True, help="Path to config file")
    parser.add("-o", "--output", type=Path, default="output", help="Path to output directory where output files will be written")
    parser.add("--cache_dir", type=Path, default="/n/holylabs/LABS/kempner_fellows/Users/jennhu/huggingface_cache/")
    parser.add("--hf_token_path", default="src/hf_token.txt", type=Path, help="Path to file containing Huggingface token.")
    
    # Model-related parameters
    parser.add("--model", type=str, default="gpt2")
    parser.add("--tokenizer", default=None, type=str)
    parser.add("--revision", default=None)
    parser.add_argument("--instruct", default=False, action="store_true")
    parser.add("--accelerate", default=False, action="store_true")
    
    # Experiment-related parameters
    parser.add("--task", type=str, choices=evaluate.EVAL_FNS.keys())
    parser.add("--condition", default=None, type=str)
    parser.add("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()
    return args

def main():
    """
    Main high-level function for running a specified experiment on a 
    specified Huggingface language model.
    """
    args = parse_args()
    print(args)

    # Initialize model.
    with open(args.hf_token_path, "r") as fp:
        token = fp.read()
    model = LM(
        args.model, 
        tokenizer_name=args.tokenizer,
        token=token,
        cache_dir=args.cache_dir, 
        revision=args.revision,
        instruct=args.instruct,
        accelerate=args.accelerate
    )

    # Set relevant keyword arguments for generation.
    kwargs = dict(max_new_tokens=args.max_new_tokens) #, do_sample=False)

    # Set path for output file.
    if args.revision is not None:
        outfile = f"{model.safe_model_name}_{args.revision}.csv"
    else:
        outfile = f"{model.safe_model_name}.csv"
    outpath = Path(args.output, args.task, outfile)
        
    # Run the evaluation.
    if args.condition is not None:
        kwargs["condition"] = args.condition
        outpath = Path(args.output, args.task, args.condition, outfile)
        
    evaluate.evaluate(
        model, 
        outpath,
        args.task,
        **kwargs
    )


if __name__ == "__main__":
    main()
