import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load the environment variables from your secure .env file
load_dotenv()

# 2. Check if the API key is present
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found!")
    exit(1)

# 3. Initialize the Groq Client
client = Groq(api_key=api_key)

print("Step 1 Complete: Classification script initialized.")
# ==========================================
# STEP 2: SYSTEM PROMPT & CATEGORY DEFINITIONS
# ==========================================

SYSTEM_PROMPT = """You are an automated customer feedback classification system. 
Your task is to read a customer review and categorize it into exactly ONE of the following allowed categories.

ALLOWED CATEGORIES & RULES:
1. Billing & Pricing - Choose this if the review mentions costs, subscriptions, charges, fees, or refunds.
2. Technical Support - Choose this if the review mentions app bugs, login issues, website crashes, or broken software features.
3. Product Quality - Choose this if the review discusses the physical item, performance, build quality, or product features.
4. Customer Service - Choose this if the review talks about shipping times, delivery packages, returning items, or support agents.

CRITICAL CONSTRAINT: You must output ONLY the exact text of the chosen category label. 
Do not include any punctuation, introductory phrases, numbers, or explanations. 
Example Output: Technical Support"""

# User template with a placeholder for the variable text input
USER_TEMPLATE = "Review Text: {REVIEW}"

print("Step 2 Complete: Categories and system rules defined.")


# ==========================================
# STEP 3: CLASSIFICATION FUNCTION
# ==========================================

def classify_review(review_text):
    """
    Fills the placeholder template and asks the LLM to classify the text
    based on the strict rules set in the system prompt.
    """
    # 1. Format the user prompt placeholder
    formatted_user_prompt = USER_TEMPLATE.format(REVIEW=review_text)
    
    # 2. Call the API using the correct system and user text paths
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": formatted_user_prompt
            }
        ],
        temperature=0.0  # 0.0 forces the model to choose the absolute best option deterministically
            # Keeps the output extremely short since we only want the category name
    )
    
    # 3. Clean up any accidental leading/trailing spaces and return the label
    return response.choices[0].message.content.strip()

print("Step 3 Complete: Classification function created.")

# ==========================================
# STEP 4: SAMPLE TEST REVIEWS & BATCH EXECUTION
# ==========================================

# 1. Define a list of test reviews representing different business issues
test_reviews = [
    "The app keeps freezing every time I try to hit the checkout button on my phone.",
    "I was charged twice on my credit card this month and I want my money back immediately!",
    "The delivery driver left my package out in the pouring rain, and customer support won't reply.",
    "The headphones sound amazing, but the plastic headband feels cheap and broke after two days."
]

print("\n--- Running Review Classification Batch Test ---")

# 2. Loop through each review and display the precise classification result
for index, review in enumerate(test_reviews, 1):
    print(f"\nReview #{index}: \"{review}\"")
    
    # Run our deterministic classifier function
    category_label = classify_review(review)
    
    # Print the clean single-label outcome
    print(f"-> Assigned Category: {category_label}")

print("\n================================================")
print("Assignment Complete: Clean batch test finalized.")
print("================================================")
