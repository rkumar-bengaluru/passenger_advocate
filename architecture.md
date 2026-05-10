%md
# Agent Architecture
![Passenger Advocate Agent Evaluation](https://raw.githubusercontent.com/rkumar-bengaluru/passenger_advocate/refs/heads/main/assets/core_architecture1.png)
**_Below is the high level description of the agent architecture. Please note actual implementation will be minimal cut down version of above architecture to save time._**

## API Gateway
**Role**: The entry point for all incoming user requests (e.g., "What's the status of my flight?").

**Function**: Receives HTTP/API calls, routes them appropriately, handles rate limiting, load balancing, and request throttling.

**Why it matters**: Provides a unified interface and protects backend services from direct exposure.

## Cache
**Role**: Stores previously computed responses or frequently accessed data.

**Function**: Before processing a new query, the system checks the cache. If a matching result exists (e.g., a recent flight status lookup for the same flight), it returns the cached answer immediately — saving time and cost.

**Why it matters**: Reduces latency and avoids redundant LLM calls or database queries.

## LangGraph Orchestrator
**Role**: The central "brain" of the agent system — coordinates the entire workflow.

**Function**: Uses LangGraph (a framework for building stateful, multi-step agent workflows) to route the query through the appropriate sequence of steps: intent classification, tool selection, data retrieval, and response generation.

**Why it matters**: Replaces rigid hardcoded pipelines with a flexible, graph-based orchestration that can branch, loop, and conditionally execute steps.

## Flight Status Query (Tool/Function Module)
**Role**: A specialized tool/module that the orchestrator can invoke to fetch real-time flight data.

**Function**: Calls airline operational databases (**hive_metastore.defaul**) to retrieve flight status information (e.g., delays, gate changes, cancellations).

**Why it matters**: Provides the factual, up-to-date data grounding that the LLM needs — avoiding hallucinations.

## Data Retrieval (RAG / Knowledge Retrieval)
**Role**: Fetches relevant knowledge from internal documents, FAQs, policy manuals, or vector databases.

**Function**: Uses Retrieval-Augmented Generation (RAG) to find contextually relevant text chunks based on the user's query, then feeds them into the LLM for answer synthesis. For the implementation i will be using openai embedding model **text-embedding-ada-002** to generate the embeddings with dimension 1536 and **chunk size** of 1024 and **chunk overlap** of 256. For storage since i already have **qdrant cluster**.

**Future**: This RAG implementation can be enhanced with **re-ranking **using bi-directional encoder followed by **cross encoder** to retrive better quality context. Please note this does degrades the performance and hence need to understand the use case better of performance is a concern and potentially discover what is the **latency** because of this.

**Why it matters**: Gives the LLM access to proprietary or domain-specific knowledge beyond its training data.

## LLM Answer Synthesis
**Role**: The language model that generates the final natural-language response.

**Function**: Takes the retrieved data (from Flight Status Query, RAG, etc.) along with the original user query and synthesizes a coherent, helpful, and empathetic response (e.g., "Your flight was delayed by 45 minutes…"). 

**Future**: Since this an MVP, i **will not attempt re-try** or **fall back to secondary model**. Personally i have gemini or gpt sometimes throws 429 errors and hence this is definately needed for a production grade agents. **My be use SLM if required on failure**.

**Why it matters**: This is the conversational AI core — it transforms raw data into human-friendly communication.

## Policy Compliance Engine
**Role**: Ensures all responses and actions comply with airline policies, regulatory requirements, and business rules.

**Function**: Reviews the generated response (or proposed action) against a ruleset — e.g., "Don't promise compensation beyond policy limits," "Don't disclose other passengers' information."

**Implementation**: I may not have this implementation for the MVP, however the idea is to check against the **guidelines**, **policies** and **guardrails** to protect **IP/PII** information leakage. **PII and compliance is a deal breaker **in few of my earlier implemntation of the agentic development.

**Why it matters**: Prevents the agent from making commitments or statements that violate company policy or regulations.

## PII Protection (Personally Identifiable Information)

(Although it seems it is repeated in compliance, because of the criticality i think i would consider this a separate component altogether)

**Role**: Detects and masks/redacts sensitive personal data.

**Function**: Scans inputs and outputs for PII (names, passport numbers, credit card details, etc.) and either redacts, masks, or encrypts them before logging or displaying.

**Why it matters**: Critical for data privacy compliance (GDPR, CCPA, etc.) and protecting passenger data.

**NOTE**: In one of the HR agent implementation with **Service Now integration**, the agent has the capability to create leave request for the user, this may contain **PHI** which needs protection along with **GDPR compliance** hacks.

##  Logging & Auditing

**Role**: Records all interactions, decisions, and data accesses.

**Function**: Creates an immutable audit trail of every query, tool call, policy check, and response — useful for debugging, compliance audits, and improvement.

**Implementation**: I have not used **Langsmith**, however i have used **Langfuse** in the agentic development, one of the drawback was either it supports masking or unmasking, which means there is no way to get the original text. We can work around the hack with **RBAC control** mechanism of masking and unmasking using **Presidio**, but this is something to keep in mind.

**Why it matters**: Essential for accountability, regulatory compliance, and continuous system improvement.


Since the implementation will be a cut down version below is the simplied flow for this agent.
![Passenger Advocate Agent Simplied Flow](https://raw.githubusercontent.com/rkumar-bengaluru/passenger_advocate/refs/heads/main/assets/simplified_flow.png)

Not all components are discussed in detail however i think the above are the major components to talk about. Below we will review the evaluation strategy which can be employed to test this agent.
