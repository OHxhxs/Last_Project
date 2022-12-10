from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer



model_name = 'smilegate-ai/kor_unsmile'

model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def filter_chatting(text):
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0,  # cpu: -1, gpu: gpu number
        return_all_scores=True,
        function_to_apply='sigmoid'
    )

    filter_text = pipe(text)[0]

    for filter in filter_text[:-1]:
        if filter['score'] > 0.6:
            return filter['label'] + "에 대한 말이 담겨있습니다. 다시 작성해주세요"

    return text


