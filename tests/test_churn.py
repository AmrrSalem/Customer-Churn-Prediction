"""
Tests for Customer Churn Prediction
Run: pytest tests/test_churn.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import engineer_features, load_telco_df


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------
@pytest.fixture
def sample_df():
    """Minimal Telco-like DataFrame for testing."""
    return pd.DataFrame({
        "customerID":       ["001", "002", "003"],
        "tenure":           [1, 24, 60],
        "MonthlyCharges":   [29.99, 59.99, 89.99],
        "TotalCharges":     [29.99, 1439.76, 5399.40],
        "Contract":         ["Month-to-month", "One year", "Two year"],
        "PaymentMethod":    ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "PhoneService":     ["Yes", "No", "Yes"],
        "InternetService":  ["DSL", "No", "Fiber optic"],
        "OnlineSecurity":   ["No", "No", "Yes"],
        "OnlineBackup":     ["Yes", "No", "Yes"],
        "DeviceProtection": ["No", "No", "Yes"],
        "TechSupport":      ["No", "No", "Yes"],
        "StreamingTV":      ["No", "No", "Yes"],
        "StreamingMovies":  ["No", "No", "Yes"],
        "Churn":            ["Yes", "No", "No"],
    })


# ----------------------------------------------------------------
# engineer_features tests
# ----------------------------------------------------------------
class TestEngineerFeatures:

    def test_returns_dataframe(self, sample_df):
        result = engineer_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        engineer_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_tenure_group_created(self, sample_df):
        result = engineer_features(sample_df)
        assert "tenure_group" in result.columns

    def test_tenure_group_values(self, sample_df):
        result = engineer_features(sample_df)
        assert result.loc[0, "tenure_group"] == "0-12"
        assert result.loc[1, "tenure_group"] == "12-24"
        assert result.loc[2, "tenure_group"] == "48+"

    def test_charges_per_tenure_no_division_error(self, sample_df):
        result = engineer_features(sample_df)
        assert "charges_per_tenure" in result.columns
        assert not result["charges_per_tenure"].isin([np.inf, -np.inf]).any()
        assert not result["charges_per_tenure"].isna().any()

    def test_monthly_to_total_ratio_created(self, sample_df):
        result = engineer_features(sample_df)
        assert "monthly_to_total_ratio" in result.columns
        assert result["monthly_to_total_ratio"].between(0, 1).all()

    def test_has_internet_and_phone(self, sample_df):
        result = engineer_features(sample_df)
        assert "has_internet_and_phone" in result.columns
        assert result.loc[0, "has_internet_and_phone"] == 1  # DSL + Yes
        assert result.loc[1, "has_internet_and_phone"] == 0  # No internet

    def test_security_and_backup(self, sample_df):
        result = engineer_features(sample_df)
        assert "security_and_backup" in result.columns
        assert result.loc[2, "security_and_backup"] == 1   # Both Yes
        assert result.loc[0, "security_and_backup"] == 0   # No security

    def test_total_services_count(self, sample_df):
        result = engineer_features(sample_df)
        assert "total_services" in result.columns
        assert result.loc[2, "total_services"] >= result.loc[0, "total_services"]

    def test_contract_risk_encoding(self, sample_df):
        result = engineer_features(sample_df)
        assert result.loc[0, "contract_risk"] == 2   # Month-to-month = high risk
        assert result.loc[1, "contract_risk"] == 1   # One year = medium
        assert result.loc[2, "contract_risk"] == 0   # Two year = low risk

    def test_electronic_check_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert result.loc[0, "is_electronic_check"] == 1
        assert result.loc[1, "is_electronic_check"] == 0

    def test_handles_zero_tenure(self):
        df = pd.DataFrame({
            "tenure": [0],
            "MonthlyCharges": [50.0],
            "TotalCharges": [0.0],
        })
        result = engineer_features(df)
        assert not result["charges_per_tenure"].isin([np.inf, -np.inf]).any()

    def test_handles_missing_columns_gracefully(self):
        df = pd.DataFrame({"tenure": [5, 30], "MonthlyCharges": [30.0, 60.0]})
        result = engineer_features(df)
        assert isinstance(result, pd.DataFrame)


# ----------------------------------------------------------------
# load_telco_df tests
# ----------------------------------------------------------------
class TestLoadTelcoDf:

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            load_telco_df("invalid_source", None)

    def test_local_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_telco_df("Local file", "/nonexistent/path/data.csv")