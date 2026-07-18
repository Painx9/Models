import streamlit as st
import pandas as pd
import os

# Configuration
DATA_FILE = "jobs.csv"

# Load or initialize data
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["Company", "Role", "Status", "Date Applied"])

st.title("🚀 Job Application Tracker")

# 1. Add new application
with st.expander("Add New Application"):
    with st.form("new_app"):
        col1, col2 = st.columns(2)
        company = col1.text_input("Company")
        role = col2.text_input("Role")
        status = st.selectbox("Status", ["Applied", "Interview", "Offer", "Rejected"])
        date_applied = st.date_input("Date Applied")
        submit = st.form_submit_button("Add")

        if submit:
            new_row = {"Company": company, "Role": role, "Status": status, "Date Applied": str(date_applied)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.rerun()

# 2. Editable table for tracking
st.subheader("Applications")
edited_df = st.data_editor(df, use_container_width=True, hide_index=True)

# 3. Save button for manual updates
if st.button("Save Changes"):
    edited_df.to_csv(DATA_FILE, index=False)
    st.success("Dashboard updated!")
