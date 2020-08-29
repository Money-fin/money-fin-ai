import sys
import torch
from kafka_helper import KafkaHelper
from config import *
from embedding import Embedding
from sentiment.models import SentimentPredictor

stop = False


def predict_sentiment_with_contents(model, sentences):
    embeddeds = []

    for sen in sentences:
        try:
            embedded = Embedding.get_sentiment_vector(sen).cuda().detach()
            embeddeds.append(embedded)
        except:
            continue
            
    embeddeds = torch.stack(embeddeds, dim=0)
    embeddeds = embeddeds.view(1, *embeddeds.size())

    pred = model(embeddeds)
    return pred.detach().cpu().numpy()[0, 0]


def predict_sentiment_with_title(model, title):
    embedded = Embedding.get_sentiment_vector(title).cuda().detach()
    pred = model(embedded.view(1, -1))
    return pred.detach().cpu().numpy()[0, 0]
    

def main():
    classification_model = None
    # sentiment_model = torch.load("sentiment/ckpts/sentiment_clf3.pt").cuda().eval()
    sentiment_model = torch.load("sentiment/ckpts/sentiment_clf-with-contents.pt").cuda().eval()

    while stop is False:
        data = KafkaHelper.consume_ninput()
        title = data["title"]
        contents = data["content"]
        link = data["link"]

        print(f"[INFO] News title: {title}")
        print(f"[INFO] News link: {link}")

        sentences = split_sentence(contents)
        pred = predict_sentiment_with_contents(sentiment_model, sentences)
        print(f"[INFO] Score: {pred:.2f}")

        # pred = predict_sentiment_with_title(sentiment_model, title)

        if pred < 0.5:
            pred = 0
        else:
            pred = 1
        print(f"[INFO] Prediction: {pred}")

        # produce message to kafka
        KafkaHelper.pub_noutput({
            "title": title,
            "link": link,
            "result": pred
        })


def split_sentence(contents):
    # print(len(contents))
    # contents = "".join(contents)
    sentences = contents.split(".")
    return sentences


if __name__ == "__main__":
    main()
