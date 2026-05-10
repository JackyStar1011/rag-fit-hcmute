import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import GENERATOR_MODEL_NAME, MAX_NEW_TOKENS


class LocalLLMGenerator:
    def __init__(self, model_name: str = GENERATOR_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading generator model: {self.model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.device == "cuda":
            # 4GB VRAM setting:
            # keep some VRAM free for CUDA overhead and KV cache
            max_memory = {
                0: "3.4GiB",
                "cpu": "12GiB",
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model.to("cpu")

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        # With device_map="auto", the first device can be cuda or cpu depending on placement.
        first_param_device = next(self.model.parameters()).device
        inputs = {key: value.to(first_param_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return answer.strip()


if __name__ == "__main__":
    from src.generation.prompt_builder import build_rag_prompt
    from src.retrieval.dense_retriever import DenseRetriever

    question = "Khoa Công nghệ Thông tin HCMUTE thành lập năm nào?"

    retriever = DenseRetriever()
    results = retriever.search(question, top_k=3)

    prompt = build_rag_prompt(question, results)

    generator = LocalLLMGenerator()
    answer = generator.generate(prompt)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)