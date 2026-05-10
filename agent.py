from langgraph import Graph, Node, Edge

# 1. Load the DAG definition (YAML file we created earlier)
graph = Graph.from_yaml("flight_status_agent.yaml")

# 2. Define tool implementations
def get_flight_status(airline_code, flight_number, flight_date):
    # Example Hive SQL query (Databricks)
    query = f"""
        SELECT Cancelled, Diverted, ArrDelayMinutes, DepDelayMinutes
        FROM hive_metastore.default.ontime_cleaned
        WHERE Reporting_Airline = '{airline_code}'
          AND Flight_Number_Reporting_Airline = '{flight_number}'
          AND FlightDate = '{flight_date}'
    """
    # Replace with actual Spark SQL execution
    result = spark.sql(query).collect()
    return result[0].asDict() if result else {}

def get_policy_context(status):
    # Replace with embedding search over policies.md
    if "Cancelled" in status:
        return "Airlines must provide refunds or rebooking options for cancelled flights."
    elif "Delayed" in status:
        return "Airlines are expected to provide timely updates and assistance for delays."
    elif "Diverted" in status:
        return "Airlines must ensure passengers reach their destination or provide alternatives."
    else:
        return "No special commitments apply for on-time flights."

def answer_synthesis(inputs):
    # Combine structured flight status with policy snippet
    cancelled = inputs.get("Cancelled", 0)
    diverted = inputs.get("Diverted", 0)
    arr_delay = inputs.get("ArrDelayMinutes", 0)
    policy = inputs.get("policy_snippet", "")

    if cancelled == 1:
        status = "Cancelled"
    elif diverted == 1:
        status = "Diverted"
    elif arr_delay > 0:
        status = f"Delayed by {arr_delay} minutes"
    else:
        status = "On time"

    return f"Flight UA123 on 2023-01-01 was {status}. {policy}"

# 3. Register implementations with the graph
graph.register_tool("get_flight_status", get_flight_status)
graph.register_tool("get_policy_context", get_policy_context)
graph.register_tool("answer_synthesis", answer_synthesis)

# 4. Execute the graph
inputs = {
    "airline_code": "UA",
    "flight_number": "123",
    "flight_date": "2023-01-01"
}

result = graph.run(inputs)
print(result["final_answer"])
