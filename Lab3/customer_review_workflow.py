"""
Two-step customer review analysis using LangChain + Groq.

Step 1: Extract structured complaint data from a raw review.
Step 2: Generate an internal support ticket using ONLY Step 1's structured data.

Install dependencies:
    pip install -U langchain-groq langchain-core

Set your Groq API key (recommended):
    Windows PowerShell:
        $env:GROQ_API_KEY="your_groq_api_key"
    Windows Command Prompt:
        set GROQ_API_KEY=your_groq_api_key

Then run:
    python customer_review_workflow.py
"""

import os
import sys
from getpass import getpass

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# You can change this to another model currently available in your Groq account.
DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_api_key() -> str:
    """Read the Groq API key from the environment, or ask securely."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        api_key = getpass("Enter your Groq API key (input hidden): ").strip()
    if not api_key:
        raise ValueError(
            "No API key provided. Set the GROQ_API_KEY environment variable "
            "or enter your key when prompted."
        )
    return api_key


def build_workflow(model_name: str):
    """Create the model, prompts, parsers, and two-step chains."""
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        api_key=get_api_key(),
    )

    json_parser = JsonOutputParser()
    text_parser = StrOutputParser()

    # STEP 1: Raw review -> structured fields.
    step_1_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert customer feedback analyst.
Analyze the customer's review and extract exactly these three fields:
- complaint: the core complaint, stated briefly
- product_or_feature: the product or feature being discussed
- sentiment: the customer's overall sentiment (Positive, Neutral, or Negative)

Return a JSON object with exactly those keys. Do not add other keys.
If a field cannot be determined, use "Unknown".
{JSON_FORMATTING}""",
            ),
            ("user", "Customer review:\n{review}"),
        ]
    ).partial(JSON_FORMATTING=json_parser.get_format_instructions())

    step_1_chain = step_1_template | llm | json_parser

    # STEP 2: Structured fields only -> internal support ticket.
    # Deliberately, this template has no {review} variable.
    step_2_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an internal customer support ticket writer.
Create a concise ticket for the internal team using ONLY the structured
fields supplied by the user message. Do not invent customer details,
causes, or facts. If a next action is useful, phrase it as a recommendation,
not as a confirmed diagnosis.

Include:
- Ticket Title
- Complaint Summary
- Product or Feature
- Customer Sentiment
- Suggested Next Action""",
            ),
            (
                "user",
                """Complaint: {complaint}
Product or Feature: {product_or_feature}
Customer Sentiment: {sentiment}""",
            ),
        ]
    )

    step_2_chain = step_2_template | llm | text_parser
    return step_1_chain, step_2_chain


def analyze_review(review: str, step_1_chain, step_2_chain) -> dict:
    """Run both steps and return both outputs."""
    if not review.strip():
        raise ValueError("Review text cannot be empty.")

    # Step 1 sees the raw review.
    structured_data = step_1_chain.invoke({"review": review})

    # Check that the expected fields exist before starting Step 2.
    required_fields = ("complaint", "product_or_feature", "sentiment")
    missing = [field for field in required_fields if field not in structured_data]
    if missing:
        raise ValueError(
            "Step 1 did not return the required fields: " + ", ".join(missing)
        )

    # Step 2 receives ONLY structured data—not the raw review.
    ticket_summary = step_2_chain.invoke(
        {
            "complaint": str(structured_data["complaint"]),
            "product_or_feature": str(structured_data["product_or_feature"]),
            "sentiment": str(structured_data["sentiment"]),
        }
    )

    return {
        "structured_complaint_data": structured_data,
        "support_ticket_summary": ticket_summary,
    }


def main():
    print("=" * 64)
    print("CUSTOMER REVIEW -> SUPPORT TICKET WORKFLOW")
    print("Step 1: Extract structured complaint data")
    print("Step 2: Create a ticket using only the structured data")
    print("=" * 64)

    model_name = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    print(f"Model: {model_name}")
    print("Tip: Set GROQ_MODEL to another model ID available in your Groq account.")
    print()

    try:
        step_1_chain, step_2_chain = build_workflow(model_name)
    except Exception as exc:
        print(f"\nCould not initialize the workflow: {exc}", file=sys.stderr)
        print(
            "Check that dependencies are installed and your Groq API key is valid.",
            file=sys.stderr,
        )
        sys.exit(1)

    while True:
        print("\nEnter a customer review. Type END on a separate line to finish:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                line = "END"
            if line.strip().upper() == "END":
                break
            lines.append(line)

        review = "\n".join(lines).strip()
        if not review:
            print("No review entered. Exiting.")
            break

        try:
            result = analyze_review(review, step_1_chain, step_2_chain)
            print("\n" + "-" * 64)
            print("STEP 1 — STRUCTURED COMPLAINT DATA")
            print("-" * 64)
            import json
            print(json.dumps(result["structured_complaint_data"], indent=2, ensure_ascii=False))

            print("\n" + "-" * 64)
            print("STEP 2 — INTERNAL SUPPORT TICKET")
            print("-" * 64)
            print(result["support_ticket_summary"])
            print("-" * 64)
        except Exception as exc:
            print(f"\nWorkflow failed: {exc}", file=sys.stderr)
            print(
                "Check your API key, model availability, internet connection, "
                "and the error details above.",
                file=sys.stderr,
            )

        again = input("\nAnalyze another review? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            break

    print("\nWorkflow finished.")


if __name__ == "__main__":
    main()
