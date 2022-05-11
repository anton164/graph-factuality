import argparse
from datasets import load_dataset
from sumtool.storage import store_model_summaries
from transformers import EncoderDecoderModel, RobertaTokenizer
from tqdm import tqdm
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_summaries(
    model,
    tokenizer,
    docs_to_summarize,
    num_beams: int = 4
):
    """
    Given a trained summary generation model and appropriate tokenizer,

    1. Tokenize text (and move to device, if possible)
    2. Run inference on model to generate output vocabulary tokens for summary
    3. Decode tokens to a sentence using the tokenizer

    Args:
        model: model to run inference on
        tokenizer: tokenizer corresponding to model
        docs_to_summarize: documents to summarize
        num_beams: number of beams for beam search

    Returns:
        decoded_sentence
    """
    inputs = tokenizer(
        docs_to_summarize,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    input_token_ids = inputs.input_ids.to(device)

    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        max_length=512,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]

    return generated_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate and store cnn daily mail test summaries using the baseline model"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = EncoderDecoderModel.from_pretrained("patrickvonplaten/roberta2roberta-cnn_dailymail-fp16")
    model.to(device)

    cnn_data = load_dataset("ccdv/cnn_dailymail", "3.0.0")["test"]
    total = len(cnn_data)
    print(str(total) + " total documents to summarize")
    i = 0
    for xs in tqdm(list(zip(*[iter(cnn_data)]*args.batch_size))):
        i += args.batch_size
        gen_summaries = generate_summaries(model, tokenizer, [x["article"] for x in xs])
        gt_summaries = [x["highlights"] for x in xs]

        store_model_summaries(
            "cnn_dailymail",
            model.config.name_or_path,
            model.config.to_dict(),
            {x["id"]: gen_summary for x, gen_summary in zip(xs, gen_summaries)},
            {x["id"]: {
                "gt_summary": x["highlights"]
            } for x in xs}
        )
        if i % 10 == 0:
            print(str(i) + " of " + str(total) + " summaries completed.")
