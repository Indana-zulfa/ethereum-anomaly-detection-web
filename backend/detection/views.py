from django.shortcuts import render
import pandas as pd

from .ml_service import predict_transaction, predict_batch


def dashboard(request):
    context = {
        "total_transactions": "920,911",
        "anomalies": "33,055",
        "normal_transactions": "887,856",
        "anomaly_rate": "3.59%",
    }

    return render(
        request,
        "detection/dashboard.html",
        context
    )


def prediction(request):

    result = None

    if request.method == "POST":

        from_address = request.POST.get("from_address")
        to_address = request.POST.get("to_address")
        value = request.POST.get("value")
        block_height = request.POST.get("block_height")
        hour = request.POST.get("hour")
        model_name = request.POST.get("model")

        result = predict_transaction(
            from_address=from_address,
            to_address=to_address,
            value=value,
            block_height=block_height,
            hour=hour,
            model_name=model_name,
        )

    return render(
        request,
        "detection/prediction.html",
        {
            "result": result
        }
    )


def batch_analysis(request):

    result = None
    error = None
    model_name = None

    if request.method == "POST":

        uploaded_file = request.FILES.get("csv_file")

        model_name = request.POST.get("model")

        # ------------------------------------------
        # Validate file
        # ------------------------------------------

        if not uploaded_file:

            error = "Please upload a CSV file."

        elif not uploaded_file.name.lower().endswith(".csv"):

            error = "Only CSV files are supported."

        elif model_name not in ["random_forest", "xgboost"]:

            error = "Please select a valid machine learning model."

        else:

            try:

                # ------------------------------------------
                # Read CSV
                # ------------------------------------------

                df = pd.read_csv(uploaded_file)


                # ------------------------------------------
                # Validate required columns
                # ------------------------------------------

                required_columns = [
                    "From",
                    "To",
                    "Value",
                    "BlockHeight",
                    "Hour",
                ]

                missing_columns = [
                    column
                    for column in required_columns
                    if column not in df.columns
                ]

                if missing_columns:

                    error = (
                        "Missing required columns: "
                        + ", ".join(missing_columns)
                    )

                elif df.empty:

                    error = "The uploaded CSV file is empty."

                else:

                    # ------------------------------------------
                    # Run batch prediction
                    # ------------------------------------------

                    result = predict_batch(
                        df,
                        model_name
                    )

                    # ------------------------------------------
                    # Summary
                    # ------------------------------------------

                    total_transactions = len(result)

                    anomaly_count = int(
                        (result["Prediction"] == 1).sum()
                    )

                    normal_count = int(
                        (result["Prediction"] == 0).sum()
                    )

                    anomaly_rate = (
                        anomaly_count / total_transactions * 100
                    )

                    summary = {
                        "total": total_transactions,
                        "anomaly": anomaly_count,
                        "normal": normal_count,
                        "anomaly_rate": anomaly_rate,
                    }


            except Exception as e:

                error = f"Error processing CSV: {str(e)}"


    context = {
        "result": result,
        "error": error,
        "model_name": model_name,
    }

    if result is not None:

        context["summary"] = summary

    return render(
        request,
        "detection/batch_analysis.html",
        context
    )


def model_information(request):

    
    return render(
        request,
        "detection/model_information.html"
    )