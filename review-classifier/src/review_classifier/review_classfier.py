import os

import openai
import pandas as pd
from dotenv import load_dotenv
from review_classifier.utils import ConfigInterface
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


def classify_reviews():
    config_interface = ConfigInterface()
    reviews_df = config_interface.load_data_from_catalog(dataset_name="movie_reviews")
    params = config_interface.load_params(key="openai")

    client = openai.Client(api_key=API_KEY)

    for i, review in tqdm(
        enumerate(reviews_df["review"]), unit="review", total=reviews_df.shape[0]
    ):
        messages = [
            {"role": "system", "content": params["system_message"]},
            {
                "role": "user",
                "content": params["human_message"].format(movie_review=review),
            },
        ]

        pred = (
            client.chat.completions.create(messages=messages, **params["kwargs"])
            .choices[0]
            .message.content
        )

        reviews_df.loc[i, "prediction"] = pred

    config_interface.save_single_dataset(
        dataset_name="sentiment_predictions", data=reviews_df.to_dict(orient="records")
    )
