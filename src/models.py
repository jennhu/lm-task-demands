import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minicons import scorer

class LM():
    """Model class for Huggingface-based LMs evaluated in our experiments."""
    def __init__(self, 
                 model_name: str, 
                 tokenizer_name=None, 
                 revision=None, 
                 accelerate=False, 
                 instruct=False,
                 **load_kwargs):
        # Store basic meta data about the model.
        self.model_name = model_name
        self.safe_model_name = self.get_file_safe_model_name(model_name)
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        else:
            self.tokenizer_name = tokenizer_name
        self.revision = revision
        self.instruct = instruct

        # Record devices, using accelerate for large models.
        if accelerate:
            self.device = "auto"
            self.input_device = 0
            print("Using accelerate")
        else:
            # Set device to GPU if cuda is available.
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Set device to CUDA")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (CUDA unvailable)")
            self.input_device = self.device

        # Initialize tokenizer and model.
        print(
            f"Initializing tokenizer ({self.tokenizer_name}) "
            f"and model ({model_name}, revision={revision})"
        )
        tokenizer, model = self.load_tokenizer_and_model(
            self.model_name, 
            self.tokenizer_name,
            revision=revision,
            accelerate=accelerate,
            **load_kwargs
        )
        self.tokenizer = tokenizer
        if accelerate:
            self.model = model
        else:
            self.model = model.to(self.device)

        # Initialize incremental LM scorer for forced-choice evaluation.
        print("Initializing incremental LM scorer")
        self.ilm_model = scorer.IncrementalLMScorer(
            self.model,
            self.device,
            tokenizer=self.tokenizer
        )

    def get_file_safe_model_name(self, model:str) -> str:
        """
        Returns a file-safe version of a Huggingface model identifier by
        only keeping the model name after a forward slash (/).
        Example: meta-llama/Llama-2-7b-hf --> Llama-2-7b-hf
        """
        safe_model_name = model.split("/")[1] if "/" in model else model
        return safe_model_name

    def load_tokenizer_and_model(self, 
                                 model_name: str, 
                                 tokenizer_name: str, 
                                 revision=None, 
                                 accelerate=False, 
                                 **kwargs):
        if revision is not None:
            kwargs["revision"] = revision
        if "allenai/OLMo" in model_name:
            import hf_olmo
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)

        if accelerate:
            # Load model using Accelerate.
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                offload_folder="offload_folder",
                torch_dtype=torch.float16,
                offload_state_dict=True,
                **kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return tokenizer, model
    
    def _prepare_inputs(self, inputs):
        tok_kwargs = dict(return_tensors="pt", padding=True)
        if "allenai/OLMo" in self.model_name:
            tok_kwargs["return_token_type_ids"] = False
        if self.instruct:
            tok_kwargs["add_special_tokens"] = False
        inputs = self.tokenizer(inputs, **tok_kwargs).to(self.input_device)
        return inputs

    def predict(self, inputs):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return logits
    
    def generate(self, input: str, **kwargs):
        inputs = self._prepare_inputs(input)
        generate_ids = self.model.generate(**inputs, **kwargs)
        output = self.tokenizer.batch_decode(
            # Subset to only keep newly generated tokens (not the input prefix)
            generate_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return output

    def get_reduction_fn(self, reduction="mean"):
        if reduction == "mean":
            reduction_fn = lambda x: x.mean(0).item()
        elif reduction == "sum":
            reduction_fn = lambda x: x.sum(0).item()
        elif reduction == "none":
            # NOTE: Typically, use 'none' for debugging purposes
            reduction_fn = lambda x: x
        else:
            raise ValueError("`reduction` should be 'mean', 'sum', or 'none'")
        return reduction_fn

    def get_logprob_of_continuation(self, 
                                    prefixes, 
                                    continuations, 
                                    separator="", 
                                    reduction="mean"):
        reduction_fn = self.get_reduction_fn(reduction=reduction)
        scores = self.ilm_model.conditional_score(
            prefixes, 
            continuations, 
            separator=separator,
            reduction=reduction_fn
        )
        return scores
    
    def get_logprob_of_sequence(self, seqs, reduction="mean"):
        reduction_fn = self.get_reduction_fn(reduction=reduction)
        scores = self.ilm_model.sequence_score(
            seqs,
            reduction=reduction_fn
        )
        return scores
