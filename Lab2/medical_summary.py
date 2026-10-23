import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load the environment variables from your secure .env file
load_dotenv()

# 2. Check if the API key is successfully found
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in your .env file!")
    exit(1)
else:
    print("✓ Success: API Key loaded correctly.")

# 3. Initialize the Groq Client
client = Groq(api_key=api_key)

print("Step 1 Complete: Client initialized successfully.")


# ==========================================
# STEP 2: SYSTEM PROMPT & TEMPLATE DESIGN
# ==========================================

# The System Prompt establishes the rules, constraints, and fixed output format
SYSTEM_PROMPT = """You are an expert medical scribe. Your task is to analyze raw clinical consultation notes and format them into a clean, standardized doctor summary.

You must ALWAYS follow this exact format structure, using these exact headings:

### Patient Name: [Insert Name]

### Symptoms
- [List symptoms as bullet points]

### Diagnosis
- [List suspected or confirmed conditions]

### Recommendation
- [List next steps, prescriptions, or advice]

CRITICAL CONSTRAINT: Do not include any introductory remarks like "Sure, here is the summary". Do not include any concluding notes. Only return the requested format sections."""

# The User Template uses placeholders to accept variable inputs dynamically
USER_TEMPLATE = """Please generate the standardized medical summary for the following consultation:

Patient Name: {NAME}
Raw Notes: {NOTES}"""

print("Step 2 Complete: System prompt and templates defined.")



# ==========================================
# STEP 3: EXECUTION FUNCTION
# ==========================================

def generate_medical_summary(patient_name, raw_notes):
    """
    Fills the placeholder template and sends the structured request 
    to Groq using system roles and a low temperature for factual accuracy.
    """
    # 1. Fill the placeholders in our template with the specific inputs
    formatted_user_prompt = USER_TEMPLATE.format(NAME=patient_name, NOTES=raw_notes)
    
    # 2. Call the API using the System and User message roles
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
        temperature=0.0  # Lowest setting ensures precise, non-creative replication of data
    )
    
    # 3. Extract and return the final text
    return response.choices[0].message.content

print("Step 3 Complete: Generation function created.")


# ==========================================
# STEP 4: SAMPLE DATA & EXECUTION
# ==========================================

# 1. Define sample messy notes from a real-world style consultation
sample_name = "Sarah Jenkins"
sample_notes = (
    "Patient came in complaining of a pounding headache on the left side for 2 days. "
    "Mentioned sensitivity to light and feeling nauseous this morning. Checked vitals—blood pressure is normal at 120/80. "
    "Likely dealing with an acute migraine episode. Advised her to rest in a dark room. "
    "Prescribed Sumatriptan 50mg to take immediately at the onset of pain. "
    "Told her to follow up if it doesn't improve in 48 hours."
)

print("\n--- Processing Raw Consultation Notes ---")

# 2. Run our generation function
final_summary = generate_medical_summary(sample_name, sample_notes)

# 3. Print the perfectly structured result
print("\n=== FINAL DOCTOR SUMMARY ===")
print(final_summary)
print("============================\n")


