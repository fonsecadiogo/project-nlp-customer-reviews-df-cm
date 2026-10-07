import os
from openai import OpenAI


def get_client():
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )


def generate_customer_insight(reviews):
    if not reviews or not reviews.strip():
        return "Please provide at least one customer review."

    prompt = f"""
You are an evidence-grounded customer review analyst.

Analyze only the customer feedback provided below.

GROUNDING RULES:
- Use only information explicitly supported by the reviews.
- Do not invent product features, problems, customer opinions, or facts.
- Do not assume an issue is common unless multiple reviews support it.
- If opinions conflict, mention the disagreement.
- Synthesize the reviews instead of simply copying them.

CUSTOMER REVIEWS:
{reviews}

TASK:
Write a concise customer insight summary.

Include:
- Overall perception
- Main strengths
- Main complaints or trade-offs
- Final takeaway
"""

    try:
        client = get_client()

        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-grounded customer review analyst. "
                        "Use only information supported by the provided reviews."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            top_p=0.95,
            max_tokens=600,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
            stream=False
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Unable to generate customer insights: {e}"