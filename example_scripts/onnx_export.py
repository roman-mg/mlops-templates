import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_id = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.use_cache = False  # this is not good option, used only for example
    model.eval()

    inputs = tokenizer("Hello world", return_tensors="pt")

    onnx_path = "outputs/qwen_model.onnx"

    torch.onnx.export(
        model,
        args=(inputs["input_ids"],),
        f=onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=16,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    main()
