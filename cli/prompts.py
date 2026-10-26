import argparse

import dotenv

from prompts.registry import load_template


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Print prompt template")
    parser.add_argument("prompt_id", help="Prompt identifier")
    parser.add_argument("version", help="Version")

    args = parser.parse_args()

    template = load_template(args.prompt_id, args.version)
    print("=== SYSTEM TEMPLATE ===")
    print(template["system"])
    print("\n=== USER TEMPLATE ===")
    print(template["user"])


if __name__ == "__main__":
    main()
