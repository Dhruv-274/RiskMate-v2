import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "RiskMate Dashboard"

# Load datasets
fraud_data = pd.read_csv("data/creditcard.csv")
credit_data = pd.read_csv("data/credit_scoring.csv")

# Rename columns for better readability
fraud_data.rename(columns={"Class": "Transaction Type"}, inplace=True)
credit_data.rename(columns={"default": "Default Status"}, inplace=True)

# Add transaction labels for simplicity
fraud_data["Transaction Label"] = fraud_data["Transaction Type"].map({0: "Legitimate", 1: "Fraudulent"})
credit_data["Default Label"] = credit_data["Default Status"].map({0: "Non-Default", 1: "Default"})

# Verify the number of non-default and default labels in the dataset
print(credit_data["Default Label"].value_counts())  # Check the distribution

# Define app layout
app.layout = html.Div(
    children=[
        html.H1("RiskMate Dashboard", style={"text-align": "center"}),

        # Fraud Detection Section
        html.Div(
            children=[
                html.H2("Fraud Detection Overview", style={"text-align": "center"}),
                dcc.Graph(
                    id="fraud-pie-chart",
                    figure=px.pie(
                        fraud_data,
                        names="Transaction Label",
                        title="Proportion of Fraudulent vs Legitimate Transactions",
                        hole=0.4,
                        color_discrete_map={"Fraudulent": "#FF5733", "Legitimate": "#28A745"}
                    )
                ),
                dcc.Graph(
                    id="fraud-bar-chart",
                    figure=px.bar(
                        fraud_data.groupby("Transaction Label")["Amount"].mean().reset_index(),
                        x="Transaction Label",
                        y="Amount",
                        title="Average Transaction Amount by Type",
                        labels={"Amount": "Average Amount", "Transaction Label": "Type"},
                        color="Transaction Label",
                        color_discrete_map={"Legitimate": "#28A745", "Fraudulent": "#FF5733"}
                    )
                )
            ],
            style={"margin-bottom": "50px", "padding": "20px", "border": "1px solid #ccc", "border-radius": "10px"}
        ),

        # Credit Scoring Section
        html.Div(
            children=[
                html.H2("Credit Risk Overview", style={"text-align": "center"}),

                # Pie Chart: Default vs Non-Default Customers
                dcc.Graph(
                    id="credit-pie-chart",
                    figure=px.pie(
                        credit_data,
                        names="Default Label",  # This ensures that we show both default and non-default labels
                        title="Proportion of Default vs Non-Default Customers",
                        hole=0.4,
                        color_discrete_map={"Non-Default": "#28A745", "Default": "#FF5733"}
                    )
                ),

                # Line Chart: Average Transaction Amount vs Transaction Frequency
                dcc.Graph(
                    id="credit-line-chart",
                    figure=px.line(
                        credit_data.groupby("transaction_frequency")["avg_transaction_amount"].mean().reset_index(),
                        x="transaction_frequency",
                        y="avg_transaction_amount",
                        title="Average Transaction Amount vs Transaction Frequency",
                        labels={
                            "transaction_frequency": "Transaction Frequency",
                            "avg_transaction_amount": "Average Amount"
                        }
                    )
                )
            ],
            style={"margin-bottom": "50px", "padding": "20px", "border": "1px solid #ccc", "border-radius": "10px"}
        ),

        # User Interaction Section
        html.Div(
            children=[
                html.H2("Interactive Insights", style={"text-align": "center"}),
                dcc.Dropdown(
                    id="transaction-dropdown",
                    options=[
                        {"label": "Legitimate Transactions", "value": "Legitimate"},
                        {"label": "Fraudulent Transactions", "value": "Fraudulent"}
                    ],
                    value="Legitimate",
                    clearable=False,
                    style={"width": "50%", "margin": "auto"}
                ),
                dcc.Graph(id="transaction-amount-distribution"),
            ],
            style={"margin-bottom": "50px", "padding": "20px", "border": "1px solid #ccc", "border-radius": "10px"}
        )
    ],
    style={"padding": "20px", "font-family": "Arial, sans-serif", "background-color": "#f4f4f9"}
)

# Callback to update transaction amount distribution chart based on dropdown selection
@app.callback(
    Output("transaction-amount-distribution", "figure"),
    Input("transaction-dropdown", "value")
)
def update_transaction_distribution(transaction_type):
    filtered_data = fraud_data[fraud_data["Transaction Label"] == transaction_type]
    return px.histogram(
        filtered_data,
        x="Amount",
        title=f"Transaction Amount Distribution: {transaction_type}",
        labels={"Amount": "Transaction Amount"},
        nbins=20,
        color_discrete_sequence=["#17becf"]
    )

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
