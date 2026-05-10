%md
# Agent Evaluation Strategy
![Passenger Advocate Agent Evaluation](https://raw.githubusercontent.com/rkumar-bengaluru/passenger_advocate/refs/heads/main/assets/eval_architecture.png)


## Passenger Advocate Agent
**Role**: The system under test. This is the actual AI agent (built with the architecture described previously) that interacts with passengers. The evaluation strategy wraps around this agent to observe its behavior.

## Test Dataset
**Role**: The input data used to "quiz" the agent.

**Function**: A curated collection of simulated passenger queries, multi-turn conversations, and edge cases (e.g., angry passengers, complex flight changes). It ensures the agent is tested against a wide variety of realistic scenarios.

## Evaluation Engine (GEval)
**Role**: The central processing unit for the evaluation.

**Function**: GEval (Generative Evaluation) is a framework that uses LLMs to evaluate other LLMs. It takes the Test Dataset, runs it through the Agent, collects the outputs, and applies the Scoring Models to calculate the various metrics. It provides a structured, step-by-step reasoning process to determine why an answer is good or bad.

## Scoring Models
**Role**: The "Judges."

**Function**: These are specialized models (often powered by advanced LLMs like GPT-4) or deterministic algorithms that actually assign scores to the agent's outputs based on the metrics defined in the diagram. They act as automated human evaluators.

## Review Dashboard
**Role**: The user interface for human overseers.

**Function**: A visual platform where developers, QA engineers, and compliance teams can view the scores, read the agent's responses, see the reasoning of the Scoring Models, and identify where the agent is failing or succeeding.

**Possible Implementation**: DeepEvals (My architechture is inspired by this framework), Ragas, langSmith, Phoenix

##  Evaluation Metrics
These are the specific KPIs (Key Performance Indicators) used to measure the agent's performance, categorized into four distinct areas:

### Quality Metrics
Measures the intrinsic quality of the agent's reasoning and final output.

**Answer Relevancy**: Does the response directly address the passenger's question, or does it include unnecessary fluff?

**Argument Correctness**: Is the logical reasoning sound? If the agent says a flight is delayed, is its justification factually correct?

**Plan Quality**: When the agent decides on a course of action (e.g., "First check flight status, then check hotel availability"), is it a good, logical plan?

**Step Efficiency**: Is the agent taking unnecessary steps? (e.g., calling an API twice when once would suffice).

**Bias**: Does the agent show unfair preference or prejudice in its responses (e.g., treating VIP passengers more politely than economy passengers)?

### Contextual Metrics
Measures how well the agent uses the provided knowledge base (RAG) and handles the overall conversation.

**Contextual Precision**: When the agent retrieves documents/policies to form an answer, are the retrieved documents highly relevant, or is there noise?

**Contextual Recall**: Did the agent retrieve all the necessary documents needed to answer the question completely?
Contextual Relevancy: Overall, is the retrieved context relevant to the user's query?

**Conversation Completeness**: Over a multi-turn chat, did the agent address all the passenger's concerns, or did it drop the ball halfway through?

**Goal Accuracy**: Did the agent actually achieve the ultimate goal the passenger set out to accomplish (e.g., successfully changing the flight)?

### Turn-Metrics
Specifically drills down into multi-turn conversations, evaluating the agent's performance at each specific step.

**Turn Contextual Precision / Recall / Relevancy**: The same as the contextual metrics above, but measured on a per-turn basis. This helps identify if the agent loses context or makes retrieval errors in the middle of a long conversation.

### Safety Metrics
Measures the risk, compliance, and safety of the agent—critical for a regulated industry like airlines.

**Hallucination**: Is the agent making up false information (e.g., inventing a fake delay reason)?

**Knowledge Retention**: Does the agent remember what was said earlier in the conversation? (e.g., If the passenger gave their booking reference in Turn 1, does the agent still know it in Turn 4?)

**PII Leakage**: Does the agent accidentally expose or reveal sensitive passenger data (Passport numbers, credit cards) that shouldn't be shown?

**Tool Use & Correctness**: Did the agent call the right tools (APIs), and did it pass the correct parameters to them? (e.g., Trying to refund a ticket instead of changing it).

**Toxicity**: Does the agent generate offensive, rude, or harmful language?

## How It All Works Together

- The Test Dataset feeds simulated passenger queries into the Passenger Advocate Agent.

- The agent executes its Agent Workflow (thinking, retrieving data, using tools).
 
- The Evaluation Engine (GEval) captures the inputs, the workflow steps, and the outputs.
 
- The Scoring Models analyze this data and calculate scores for all the Quality, Contextual, Turn, and Safety Metrics.
 
- The results are sent to the Review Dashboard, allowing engineers to continuously monitor and improve the agent before and after it interacts with real passengers.