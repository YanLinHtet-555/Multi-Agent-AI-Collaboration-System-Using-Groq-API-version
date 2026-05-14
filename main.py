#!/usr/bin/env python3
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


EXAMPLE_QUERIES = [
    "Build a Python REST API with FastAPI that manages a todo list with CRUD operations",
    "Research the best practices for securing JWT tokens and implement a Python JWT utility",
    "Create a Python web scraper that extracts product prices from an e-commerce page",
    "Design and implement a rate limiter in Python using the token bucket algorithm",
]


def check_env() -> bool:
    if not os.getenv("GROQ_API_KEY"):
        print(
            "Error: GROQ_API_KEY environment variable is not set.\n"
            "Get a free key at: https://console.groq.com/keys\n"
            "Then add to .env: GROQ_API_KEY=your_key_here"
        )
        return False
    return True


def run_query(query: str, verbose: bool = True) -> str:
    from agents import ManagerAgent

    manager = ManagerAgent()
    return manager.orchestrate(query, verbose=verbose)


def interactive_mode(verbose: bool) -> None:
    print("=" * 60)
    print("  Multi-Agent AI Collaboration System (Groq)")
    print("=" * 60)
    print("Type your query and press Enter. Type 'quit' to exit.")
    print("Type 'examples' to see sample queries.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "examples":
            print("\nExample queries:")
            for i, q in enumerate(EXAMPLE_QUERIES, 1):
                print(f"  {i}. {q}")
            print()
            continue

        print()
        result = run_query(query, verbose=verbose)
        print("\n" + "=" * 60)
        print("FINAL RESULT:")
        print("=" * 60)
        print(result)
        print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI Collaboration System powered by Groq"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose agent activity logs",
    )
    args = parser.parse_args()

    if not check_env():
        sys.exit(1)

    verbose = not args.quiet

    if args.query:
        result = run_query(args.query, verbose=verbose)
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(result)
    else:
        interactive_mode(verbose=verbose)


if __name__ == "__main__":
    main()
