import os

import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from movie_review_classifier import settings

configure_project(settings.BASE_PATH)
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


def classify_reviews():
    with KedroSession.create(project_path=settings.PROJECT_PATH, env="base") as session:
        context = session.load_context()
        reviews_df = context.catalog.load("movie_reviews")
        openai_params = context.catalog.load("parameters")["openai"]

    client = openai.Client(api_key=API_KEY)

    for i, review in tqdm(
        enumerate(reviews_df["review"]), unit="review", total=reviews_df.shape[0]
    ):
        messages = [
            {"role": "system", "content": openai_params["system_message"]},
            {
                "role": "user",
                "content": openai_params["human_message"].format(movie_review=review),
            },
        ]

        pred = (
            client.chat.completions.create(messages=messages, **openai_params["kwargs"])
            .choices[0]
            .message.content
        )

        reviews_df.loc[i, "prediction"] = pred

    context.catalog.save(
        name="sentiment_predictions", data=reviews_df.to_dict(orient="records")
    )
