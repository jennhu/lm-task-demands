import os

def make_yaml(task, model, accelerate=False, instruct=False, condition=None):
    yml = f"""
task: {task}
model: {model}"""
    if condition is not None:
        yml += f"\ncondition: {condition}"
    if accelerate:
        yml += f"\naccelerate: true"
    if instruct:
        yml += f"\ninstruct: true"

    if task == "digit_mat":
        yml += f"\nmax_new_tokens: 10"
    else:
        yml += f"\nmax_new_tokens: 100"
    return yml

def get_file_safe_model_name(model):
    return model.split("/")[-1]

def write_to_file(yml, out_path):
    with open(out_path, "w") as fp:
        fp.write(yml)
    print(f"Wrote file to {out_path}")

def write_config(task, model, output_folder, **kwargs):
    yml = make_yaml(task, model, **kwargs)
    safe_model_name = get_file_safe_model_name(model)
    write_to_file(yml, f"{output_folder}/{safe_model_name}.yaml")

def main(config_folder="configs"):
    models = [
        # normal models
        ("google/gemma-2b", False, False),
        ("google/gemma-7b", False, False),
        ("mistralai/Mistral-7B-v0.1", False, False),
        ("allenai/OLMo-1B", False, False),
        ("allenai/OLMo-7B", False, False),
        ("meta-llama/Llama-2-7b-hf", False, False),
        ("EleutherAI/pythia-1b-deduped", False, False),
        ("EleutherAI/pythia-1.4b-deduped", False, False),
        ("EleutherAI/pythia-2.8b-deduped", False, False),
        ("EleutherAI/pythia-6.9b-deduped", False, False),
        # models that need accelerate
        ("meta-llama/Llama-2-13b-hf", True, False),
        ("EleutherAI/pythia-12b-deduped", True, False),
        ("meta-llama/Llama-2-70b-hf", True, False),
    ]

    tasks = {
        "lambada": [],
        "crt": ["crt1", "crt2", "crt3"],
        "digit_mat": [],
        "dgl": [],
        "blimp": []
    }

    for task, conditions in tasks.items():
        if len(conditions) > 0:
            for condition in conditions:
                output_folder = f"{config_folder}/{task}/{condition}"
                os.makedirs(output_folder, exist_ok=True)
                for model, accelerate, instruct in models:
                    write_config(
                        task, 
                        model, 
                        output_folder, 
                        accelerate=accelerate, 
                        condition=condition,
                        instruct=instruct
                    )
        else:
            output_folder = f"{config_folder}/{task}"
            os.makedirs(output_folder, exist_ok=True)
            for model, accelerate, instruct in models:
                write_config(
                    task, 
                    model, 
                    output_folder, 
                    accelerate=accelerate, 
                    condition=None,
                    instruct=instruct
                )


if __name__ == "__main__":
    main()