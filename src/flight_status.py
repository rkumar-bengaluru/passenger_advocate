# from pyspark.sql import SparkSession
from datetime import datetime
import json
from typing import Dict, Any

from pydantic import BaseModel, Field

# spark = SparkSession.builder.getOrCreate()


class GetFlightStatusInput(BaseModel):
    airline: str = Field(..., description="Airline Name for which the information needs to be retrieved")
    flight_number: str = Field(..., description="Flight for which the status required")
    flight_date: str = Field(..., description="Date of the flight")

def get_flight_status(
    airline: str,
    flight_number: int,
    flight_date: str
) -> Dict[str, Any]:
    """
    Get flight status from ontime_cleaned table using Spark.
    Works directly in your Databricks notebook.
    """
    
    # Normalize inputs
    airline = airline.strip().upper()
    # Convert YYYY-MM-DD to yyyymmdd format (most common in this dataset)
    flight_date_hive = flight_date.replace("-", "")
    
    query = f"""
    SELECT 
        FlightDate,
        Reporting_Airline,
        Flight_Number_Reporting_Airline,
        Origin,
        OriginCityName,
        Dest,
        DestCityName,
        CRSDepTime,
        DepTime,
        DepDelayMinutes,
        CRSArrTime,
        ArrTime,
        ArrDelayMinutes,
        Cancelled,
        CancellationCode,
        Diverted,
        DivAirportLandings,
        ActualElapsedTime,
        AirTime,
        CarrierDelay,
        WeatherDelay,
        NASDelay,
        SecurityDelay,
        LateAircraftDelay
    FROM hive_metastore.default.ontime_cleaned
    WHERE Reporting_Airline = '{airline}'
      AND Flight_Number_Reporting_Airline = {flight_number}
      AND FlightDate = '{flight_date_hive}'
    ORDER BY Origin, Dest
    """

    try:
        # df = spark.sql(query)
        # results = [row.asDict() for row in df.collect()]
        
        # # Create human-readable summary
        # summary = None
        # if results:
        #     flight = results[0]
        #     if flight.get('Cancelled') == 1:
        #         summary = f"Cancelled - {flight.get('CancellationCode', '')}"
        #     elif flight.get('Diverted') == 1:
        #         summary = "Diverted"
        #     elif flight.get('ArrDelayMinutes', 0) >= 15:
        #         summary = f"Delayed by {flight.get('ArrDelayMinutes')} minutes"
        #     else:
        #         summary = "On Time"
        results = []
        summary = "Delayed by 35 minutes"
        response = {
            "tool": "get_flight_status",
            "input": {
                "airline": airline,
                "flight_number": flight_number,
                "flight_date": flight_date
            },
            "query_executed": query.strip(),
            "status": "success",
            "record_count": len(results),
            "data": results,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        return {
            "tool": "get_flight_status",
            "input": {"airline": airline, "flight_number": flight_number, "flight_date": flight_date},
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    result = get_flight_status("UA", 123, "2023-01-01")
    print(json.dumps(result, indent=2, default=str))