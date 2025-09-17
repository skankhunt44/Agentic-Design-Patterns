from google.adk.agents import SequentialAgent, LlmAgent

# The first agent generates the initial draft.
generator = LlmAgent(
   name="DraftWriter",
   description="Generates initial draft content on a given subject.",
   instruction="Write a short, informative paragraph about the user's subject.",
   output_key="draft_text" # The output is saved to this state key.
)

# The second agent critiques the draft from the first agent.
reviewer = LlmAgent(
   name="FactChecker",
   description="Reviews a given text for factual accuracy and provides a structured critique.",
   instruction="""
   You are a meticulous fact-checker.
   1. Read the text provided in the state key 'draft_text'.
   2. Carefully verify the factual accuracy of all claims.
   3. Your final output must be a dictionary containing two keys:
      - "status": A string, either "ACCURATE" or "INACCURATE".
      - "reasoning": A string providing a clear explanation for your status, citing specific issues if any are found.
   """,
   output_key="review_output" # The structured dictionary is saved here.
)

# The SequentialAgent ensures the generator runs before the reviewer.
review_pipeline = SequentialAgent(
   name="WriteAndReview_Pipeline",
   sub_agents=[generator, reviewer]
)

# Execution Flow:
# 1. generator runs -> saves its paragraph to state['draft_text'].
# 2. reviewer runs -> reads state['draft_text'] and saves its dictionary output to state['review_output'].
