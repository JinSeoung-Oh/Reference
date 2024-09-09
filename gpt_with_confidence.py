import math
from openai import OpenAI

client = OpenAI()

movie_name = "Gladiator"

genres = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "War", "Western"]
genre_string = "\n".join([f"{i}. {g}" for i, g in enumerate(genres)])

prompt = f"""\
Which genre best describes the movie {movie_name!r}?
Consider a few likely genres and explain your reasoning, 
then pick an answer from the list below 
and show it in answer tags, like: <answer>4</answer>
{genre_string}
"""

# Call the API, requesting logprobs and 10 top_logprobs
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[dict(role="user", content=prompt)],
    logprobs=True,
    top_logprobs=10,
)

# Extract the responses and confidences
label_dict = {}
text = ""
for tokenLogProb in completion.choices[0].logprobs.content:
    # When we get to the token following '<answer>', extract alternatives listed in top_logprobs
    if text.endswith("<answer>"):
        for item in tokenLogProb.top_logprobs:
            if (confidence := math.exp(item.logprob)) > 0.01:
                genre = genres[int(item.token)]
                label_dict[genre] = confidence
    text += tokenLogProb.token


for genre, confidence in label_dict.items():
    print(f"{genre}: {confidence:.2%}")
