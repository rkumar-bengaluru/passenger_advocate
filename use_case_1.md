%md
# Use case 
## My flight was delayed 3 hours. Do I get a meal voucher?

### Agent Build
```python


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_classification", intent_classification)
    workflow.add_node("get_flight_status", flight_status_query)
    workflow.add_node("get_policy_details", get_policy_details)
    workflow.add_node("summarize_answer", summarize_answer)
    workflow.add_node("policy_guardrails", policy_guardrails)
    workflow.add_node("pii_check", pii_check)

    # Start → Intent
    workflow.add_edge(START, "intent_classification")

    # Conditional routing from intent
    workflow.add_conditional_edges(
        "intent_classification",
        route_from_intent_from_analysis,
        {
            "get_flight_status": "get_flight_status",
            "get_policy_details": "get_policy_details",
            "summarize_answer": "summarize_answer"
        }
    )

    # Downstream edges
    workflow.add_edge("get_flight_status", "summarize_answer")
    workflow.add_edge("get_policy_details", "summarize_answer")
    workflow.add_edge("summarize_answer", "policy_guardrails")
    workflow.add_edge("policy_guardrails", "pii_check")
    workflow.add_edge("pii_check", END)

    return workflow.compile()

```

### RAG Context
    [1] (score=0.8439): ### Frontier Airlines’ Commitments for Controllable Delays
*   Rebook passenger on same airline at no additional cost for significant delays
*   Meal or meal cash/voucher when flight delay results in passenger waiting for 3 hours or more

Frontier Airlines does not commit to:
*   Rebook on partner airline or another airline with which it has an agreement at no additional cost for significant delays
*   Complimentary hotel accommodations for any passenger affected by an overnight delay
*   Complimentary ground transportation to and from hotel for any passenger affected by an overnight delay
*   Cash compensation when a delay results in passenger waiting for 3 hours or more from the scheduled departure time
*   Credit/travel voucher when delay results in passenger waiting for 3 hours or more from the scheduled departure time
*   Frequent flyer miles when delay results in passenger waiting for 3 hours or more from the scheduled departure time
[2] (score=0.8336): JetBlue Airways does not commit to:
*   Cash compensation when a delay results in passenger waiting for 3 hours or more from the scheduled departure time
*   Frequent flyer miles when delay results in passenger waiting for 3 hours or more from the scheduled departure time

### Obserbility and Monitoring Logs in Langfuse
![Passenger Advocate Agent Evaluation1](https://raw.githubusercontent.com/rkumar-bengaluru/passenger_advocate/refs/heads/main/assets//user_case_1_monitoring.png)

### Final Output
Agent: Whether you are entitled to a meal voucher depends on the airline and the cause of the delay.

For a **controllable delay** of 3 hours or more, the following airlines commit to providing a meal or a meal voucher:
*   Delta Air Lines [3]
*   Frontier Airlines [1]
*   JetBlue Airways [5]
*   Southwest Airlines [2]
*   United Airlines [4]
