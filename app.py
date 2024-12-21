import streamlit as st
from pyresparser import ResumeParser
from bs4 import BeautifulSoup
import pandas as pd
import requests
from rapidfuzz import fuzz
import smtplib
from email.mime.text import MIMEText
import nltk
import os

# ----------------- NLTK Setup ----------------- #
def setup_nltk():
    """Initialize NLTK and download required data."""
    try:
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = '/home/vscode/nltk_data'
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', download_dir=nltk_data_dir)
            nltk.download('punkt', download_dir=nltk_data_dir)
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
            nltk.download('universal_tagset', download_dir=nltk_data_dir)
            st.success("Successfully downloaded NLTK data!")
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        raise e


# ----------------- Job Scraping Functions ----------------- #

# Seek Job Scraper
def get_seek_jobs(keyword, location, country):
    country_domains = {
        "Australia": "seek.com.au",
        "United States": "seek.com/us",
        "United Kingdom": "seek.co.uk",
        "Canada": "seek.co.ca",
        "India": "seek.co.in",
    }
    domain = country_domains.get(country)
    if not domain:
        return []
    base_url = f"https://www.{domain}/{keyword}-jobs/in-{location}"
    response = requests.get(base_url)
    jobs = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        for job in soup.find_all('div', class_='card'):
            title = job.find('a', class_='job-title').text.strip()
            description = job.find('div', class_='job-description').text.strip()
            jobs.append({'title': title, 'description': description})
    return jobs

# Indeed RSS Feed
def get_indeed_jobs(keyword, location, country):
    country_domains = {
        "Australia": "au",
        "United States": "us",
        "United Kingdom": "uk",
        "Canada": "ca",
        "India": "in",
        "Germany": "de",
        "France": "fr",
        "Singapore": "sg",
        "United Arab Emirates": "ae",
        "Japan": "jp",
    }
    domain = country_domains.get(country)
    if not domain:
        return []
    rss_url = f"https://{domain}.indeed.com/rss?q={keyword}&l={location}"
    response = requests.get(rss_url)
    jobs = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'xml')
        jobs = [{'title': item.title.text, 'description': item.description.text}
                for item in soup.find_all('item')]
    return jobs

# Monster RSS Feed
def get_monster_jobs(keyword, location, country):
    rss_url = f"https://rss.jobsearch.monster.com/rssquery.ashx?q={keyword}&where={location}&country={country}"
    response = requests.get(rss_url)
    jobs = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'xml')
        jobs = [{'title': item.title.text, 'description': item.description.text}
                for item in soup.find_all('item')]
    return jobs

# Adzuna API
def get_adzuna_jobs(api_key, keyword, location, country_code):
    url = f"http://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
    params = {'app_id': 'YOUR_ADZUNA_APP_ID', 'app_key': api_key, 'what': keyword, 'where': location}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return [{'title': job['title'], 'description': job['description']}
                for job in response.json()['results']]
    return []


# ----------------- Resume Parsing ----------------- #
def parse_resume(file_path):
    resume_data = ResumeParser(file_path).get_extracted_data()
    return resume_data.get('skills', [])

# ----------------- Job Matching ----------------- #
def calculate_match(job_description, resume_skills):
    skills_str = " ".join(resume_skills)
    return fuzz.partial_ratio(skills_str.lower(), job_description.lower())

# ----------------- Email Notifications ----------------- #
def send_email_notification(jobs, recipient_email):
    sender_email = "your_email@gmail.com"
    password = "your_email_password"

    # Email content
    body = "\n\n".join([f"Title: {job['title']}\nDescription: {job['description']}" for job in jobs])
    msg = MIMEText(body)
    msg['Subject'] = "Top Matched Jobs"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Send email using Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)

# ----------------- Streamlit UI ----------------- #
def main():
    # Setup NLTK first
    setup_nltk()
    st.title("Global AI/ML Job Matching Assistant")
    st.sidebar.header("Upload Resume & Preferences")

    # Inputs
    uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_keyword = st.sidebar.text_input("Job Keyword", "AI/ML")
    job_location = st.sidebar.text_input("Job Location", "City")
    country = st.sidebar.selectbox("Country", [
        "Australia", "United States", "United Kingdom", "Canada", "India",
        "Germany", "France", "Singapore", "United Arab Emirates", "Japan"
    ])
    recipient_email = st.sidebar.text_input("Your Email for Notifications", "")       

    if uploaded_file and st.sidebar.button("Find Jobs"):
        # Parse Resume
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        resume_skills = parse_resume("temp_resume.pdf")
        st.success(f"Skills Extracted: {', '.join(resume_skills)}")

        # Fetch Jobs
        st.info("Fetching Jobs from Seek, Indeed, Monster, and Adzuna...")
        jobs = []
        jobs += get_seek_jobs(job_keyword, job_location, country)
        jobs += get_indeed_jobs(job_keyword, job_location, country)
        jobs += get_monster_jobs(job_keyword, job_location, country)

        # Match Jobs
        matched_jobs = [
            job for job in jobs if calculate_match(job['description'], resume_skills) > 60
        ]
        
        # Display Jobs
        if matched_jobs:
            st.subheader("Top Matched Jobs:")
            for job in matched_jobs:
                st.write(f"**{job['title']}**")
                st.write(job['description'])
                st.write("---")

            # Email Results
            if recipient_email:
                send_email_notification(matched_jobs, recipient_email)
                st.success(f"Top matches sent to {recipient_email}!")

            # Download Results
            df = pd.DataFrame(matched_jobs)
            df.to_csv("matched_jobs.csv", index=False)
            st.download_button("Download Matched Jobs as CSV", "matched_jobs.csv", "text/csv")
        else:
            st.warning("No suitable jobs found. Try adjusting your filters.")

if __name__ == "__main__":
    main()
