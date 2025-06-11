import json
from openai import OpenAI

client = OpenAI(api_key="")


def extract_items_from_text(text):
    functions = [
        {
            "name": "extract_quote_items",
            "description": "Extracts itemized data from a quote or invoice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Request Item": {"type": "string"},
                                "Quantity": {"type": ["integer", "null"]},
                                "Unit Price": {"type": ["number", "null"]},
                                "Total": {"type": ["number", "null"]}
                            },
                            "required": ["Request Item", "Quantity"]
                        }
                    }
                },
                "required": ["items"]
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Extract all itemized product data from the following text:\n\n{text}"
            }
        ],
        tools=[{"type": "function", "function": functions[0]}],
        tool_choice={"type": "function", "function": {"name": "extract_quote_items"}},
        temperature=0
    )

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        parsed = json.loads(args)
        return parsed["items"]
    except Exception as e:
        raise ValueError("Failed to parse tool_call arguments as JSON:\n" + str(e)) from e

