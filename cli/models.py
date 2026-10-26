import argparse
import asyncio
import json
import sys
from typing import Any

import dotenv

import models


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Test the sdbench model chat completion functionality directly"
    )
    parser.add_argument("model", help="Model to use (e.g., 'o4-mini', 'claude-4-sonnet', 'gpt-4o')")
    parser.add_argument("user_message", help="User message to send to the model")
    parser.add_argument("--system", help="System message (optional)")
    parser.add_argument(
        "--model-args",
        type=str,
        help='Additional model arguments as JSON string (e.g., \'{"temperature": 0.8, "max_tokens": 100}\')',
    )
    parser.add_argument(
        "--output-format",
        choices=["content", "raw"],
        default="content",
        help="Output format: 'content' (response message only), 'raw' (full response output)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to show request details"
    )
    parser.add_argument(
        "--async",
        action="store_true",
        dest="run_async",
        help="Use async version of create_chat_completion",
    )

    args = parser.parse_args()

    messages: list[models.ChatCompletionMessageParam] = []

    if args.system:
        messages.append({"role": "system", "content": args.system})

    messages.append({"role": "user", "content": args.user_message})

    # Parse model arguments if provided
    model_args: dict[str, Any] = {}
    if args.model_args:
        try:
            model_args = json.loads(args.model_args)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --model-args: {e}")
            sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Model Args: {json.dumps(model_args, indent=2)}")
    print("Messages:")
    models.print_messages(messages)
    print()

    if args.run_async:
        print(f"Calling model '{args.model}' (async)...")
        response = asyncio.run(
            models.create_chat_completion_async(
                model=args.model, messages=messages, debug=args.debug, **model_args
            )
        )
    else:
        print(f"Calling model '{args.model}'...")
        response = models.create_chat_completion(
            model=args.model, messages=messages, debug=args.debug, **model_args
        )

    # Format and display output
    if args.output_format == "content":
        models.print_messages(response.choices[0].message)

    elif args.output_format == "raw":
        print("=== RESPONSE DETAILS ===")
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
