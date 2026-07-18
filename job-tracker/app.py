import streamlit as st
import pandas as pd
import os

# Configuration
DATA_FILE = "jobs.csv"

# Load or initialize data (handling the new Link column seamlessly)
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    # Ensure Link column exists if loading an older CSV file
    if "Link" not in df.columns:
        df["Link"] = ""
else:
    df = pd.DataFrame(columns=["Company", "Role", "Status", "Date Applied", "Link"])

st.title("🚀 Job Application Tracker")

# 1. Add new application
with st.expander("Add New Application"):
    with st.form("new_app"):
        col1, col2 = st.columns(2)
        company = col1.text_input("Company")
        role = col2.text_input("Role")
        
        col3, col4 = st.columns(2)
        status = col3.selectbox("Status", ["Applied", "Interview", "Offer", "Rejected"])
        date_applied = col4.date_input("Date Applied")
        
        link = st.text_input("Job Posting Link (URL)", placeholder="https://...")
        submit = st.form_submit_button("Add Application")

        if submit:
            new_row = {
                "Company": company, 
                "Role": role, 
                "Status": status, 
                "Date Applied": str(date_applied),
                "Link": link
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.rerun()

# 2. Editable table with clickable links
st.subheader("Applications")

# Configure the table columns to render URLs cleanly
column_config = {
    "Link": st.column_config.LinkColumn(
        "Job Link",
        help="Click to open the job posting",
        placeholder="No URL provided",
        display_text="Open Link"  # <-- This ensures Streamlit renders the text correctly
    )
}

edited_df = st.data_editor(
    df, 
    use_container_width=True, 
    hide_index=True,
    column_config=column_config
)

# 3. Save changes automatically if edited in the table
if st.button("Save Changes"):
    edited_df.to_csv(DATA_FILE, index=False)
    st.success("Dashboard updated successfully!")
