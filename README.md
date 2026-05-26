# A quest in the clouds

### Q. What is this quest?
It is a fun way to assess your **AI Engineering** skills. It is also a representative sample of the work we do at Rearc.

We've built a basic orchestrator with Langgraph. Your job is to build a **Passenger Advocate Agent** that can help travelers understand their rights when things go wrong at the airport.

### Q. So what skills should I have?
- **Python proficiency:** You can write effective, modular code.
- **LLM API Usage:** Experience with OpenAI, Anthropic, or AWS Bedrock APIs.
- **Orchestration:** Understanding of state management and agentic workflows (e.g., LangGraph, State Machines, or custom control flows).
- **Tool Use:** Ability to define structured interfaces for LLMs to interact with data.
- **RAG concepts:** Retrieval, context injection, and semantic search.

### Q. What to do?
We have provided two data sources:
1.  `dataset`: A dataset of flight records, delays, and cancellations.(https://www.kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024)
2.  `policies.md`: The official "Consumer Guide to Air Travel," including specific commitments from airlines regarding cancellations and delays.

**The Challenge:**
Build a Python application where an LLM Agent acts as a Travel Advocate. The Agent must answer user questions by determining *which* data source to use, *how* to query it, and *how* to synthesize the final answer.

### Q. How do I know I have solved it?
Your Agent should be able to answer the following types of questions correctly:

1.  **User:** *"What was the status of flight UA123 on 2023-01-01?"*
2.  **User:** *"Since it was cancelled, is United required to provide me with a hotel?"*
3.  **User:** *"My flight was delayed 3 hours. Do I get a meal voucher?"*

