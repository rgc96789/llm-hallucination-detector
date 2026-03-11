import openai

def ask_model(question):

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":question}]
    )

    return response["choices"][0]["message"]["content"]


def evaluate_hallucination(question, answer):

    prompt = f"""
You are an AI evaluation system.

Question:
{question}

Answer:
{answer}

Evaluate the risk of hallucination.

Score from 1–10 for:
- factual accuracy
- likelihood of hallucination
- confidence level

Explain reasoning briefly.
"""

    evaluation = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}]
    )

    return evaluation["choices"][0]["message"]["content"]


if __name__ == "__main__":

    question = "Who invented calculus?"

    answer = ask_model(question)

    print("MODEL ANSWER:\n", answer)

    result = evaluate_hallucination(question, answer)

    print("\nEVALUATION:\n", result)
