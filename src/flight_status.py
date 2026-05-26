from pyspark.sql import SparkSession
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
    flight_number: str,
    flight_date: str
) -> Dict[str, Any]:
    """
    Get flight status from ontime_cleaned table using Spark.
    Works directly in your Databricks notebook.
    """
    from datetime import datetime

    # Normalize inputs
    airline = airline.strip().upper()
    # Keep YYYY-MM-DD format (matches Hive table)
    flight_date_hive = flight_date.strip()

    print(f"""
    --- Flight Query Parameters ---
    Airline Code     : {airline}
    Flight Number    : {flight_number}
    Flight Date (raw): {flight_date}
    Flight Date Hive : {flight_date_hive}
    --------------------------------
    """)

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
      AND Flight_Number_Reporting_Airline = {int(flight_number)}
      AND FlightDate = '{flight_date_hive}'
    ORDER BY Origin, Dest
    """

    try:

        summary = f"""Flight Status Summary:
        Flight {airline} {flight_number} from Salt Lake City to Cedar City on {flight_date_hive} was CANCELLED
        """

        # df = spark.sql(query)
        # results = [row.asDict() for row in df.collect()]

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

        return {
            "tool": "get_flight_status",
            "input": {"airline": airline, "flight_number": flight_number, "flight_date": flight_date},
            "status": "success",
            "summary": summary
        }

    except Exception as e:
        return {
            "tool": "get_flight_status",
            "input": {"airline": airline, "flight_number": flight_number, "flight_date": flight_date},
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
