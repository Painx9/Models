import streamlit as st
import pandas as pd
import os

# Configuration
DATA_FILE = "jobs.csv"

# Load or initialize data
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
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
            # We save clean raw text urls in the CSV, but format them dynamically
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

# 2. Editable table using TextColumn with Markdown links
st.subheader("Applications")

# Duplicate our DataFrame for display purposes so we can transform the raw URLs into clean Markdown strings
display_df = df.copy()

def format_markdown_link(url):
    if pd.isna(url) or str(url).strip() == "" or not str(url).startswith("http"):
        return "⚠️ No Link"
    return f"[🔗 Open Link]({url})"

display_df["Link"] = display_df["Link"].apply(format_markdown_link)

# We use the highly stable Markdown TextColumn config instead of LinkColumn to evade the Python 3.14 bug
column_config = {
    "Link": st.column_config.TextColumn(
        "Job Link",
        help="Click the Markdown link to open the posting",
        disabled=False # Keeps the field editable if you need to paste a fresh text URL
    )
}

edited_df = st.data_editor(
    display_df, 
    use_container_width=True, 
    hide_index=True,
    column_config=column_config
)

# 3. Save changes cleanly
if st.button("Save Changes"):
    # Before saving back to the CSV, we extract the raw text out of the markdown wrapper if edited
    def extract_raw_url(markdown_text):
        if "](" in str(markdown_text):
            return str(markdown_text).split("](")[1].replace(")", "")
        return markdown_text

    edited_df["Link"] = edited_df["Link"].apply(extract_raw_url)
    edited_df.to_csv(DATA_FILE, index=False)
    st.success("Dashboard updated successfully!")
