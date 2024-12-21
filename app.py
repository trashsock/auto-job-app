import os
import nltk
import spacy
import streamlit as st
from bs4 import BeautifulSoup
import pandas as pd
import requests
from rapidfuzz import fuzz
import smtplib
from email.mime.text import MIMEText
import pdfplumber
import re

# ----------------- Setup and Initialization ----------------- #
def setup_nlp():
    """Initialize NLP components"""
    # Download NLTK data
    nltk_data_dir = '/home/vscode/nltk_data'
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    for item in ['stopwords', 'punkt']:
        nltk.download(item, download_dir=nltk_data_dir, quiet=True)
    
    # Load spaCy model
    try:
        nlp = spacy.load('en_core_web_sm')
        return nlp
    except OSError:
        st.warning("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load('en_core_web_sm')

# ----------------- Resume Parsing ----------------- #
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_skills(text, nlp):
    """Extract skills from text using spaCy"""
    # Common technical skills
    SKILLS_DB = {
        'programming': [
            'python', 'java', 'javascript', 'c++', 'ruby', 'php', 'sql', 'r', 
            'matlab', 'scala', 'swift', 'golang'
        ],
        'frameworks': [
            'django', 'flask', 'react', 'angular', 'vue', 'node.js', 'express',
            'spring', 'tensorflow', 'pytorch', 'keras'
        ],
        'databases': [
            'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'elasticsearch',
            'cassandra', 'sqlite'
        ],
        'tools': [
            'git', 'docker', 'kubernetes', 'jenkins', 'jira', 'aws', 'azure',
            'gcp', 'linux', 'bash', 'rest api', 'graphql'
        ],
        'machine_learning': [
            'machine learning', 'deep learning', 'neural networks', 'nlp',
            'computer vision', 'data science', 'ai', 'artificial intelligence',
            'data mining', 'data analysis', 'big data'
        ]
    }
    
    # Flatten skills list
    all_skills = [skill for category in SKILLS_DB.values() for skill in category]
    
    # Process text with spaCy
    doc = nlp(text.lower())
    
    # Extract skills
    found_skills = set()
    for token in doc:
        if token.text in all_skills:
            found_skills.add(token.text)
        # Check for compound skills (e.g., "machine learning")
        if token.i < len(doc) - 1:
            bigram = token.text + ' ' + doc[token.i + 1].text
            if bigram in all_skills:
                found_skills.add(bigram)
    
    return list(found_skills)

def parse_resume(file_path, nlp):
    """Parse resume and extract relevant information"""
    text = extract_text_from_pdf(file_path)
    skills = extract_skills(text, nlp)
    return {
        'skills': skills,
        'raw_text': text
    }

# ----------------- Job Scraping Functions ----------------- #
def get_seek_jobs(keyword, location, country):
    """Fetch jobs from Seek with error handling"""
    try:
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
        response = requests.get(base_url, verify=True, timeout=10)
        jobs = []
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for job in soup.find_all('div', class_='card'):
                try:
                    title = job.find('a', class_='job-title').text.strip()
                    description = job.find('div', class_='job-description').text.strip()
                    jobs.append({
                        'source': 'Seek',
                        'title': title,
                        'description': description
                    })
                except AttributeError:
                    continue
        return jobs
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching jobs from Seek: {str(e)}")
        return []

def get_indeed_jobs(keyword, location, country):
    """Fetch jobs from Indeed with error handling"""
    try:
        country_domains = {
            "Australia": "au",
            "United States": "com",
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
        response = requests.get(rss_url, verify=True, timeout=10)
        jobs = []
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            for item in soup.find_all('item'):
                jobs.append({
                    'source': 'Indeed',
                    'title': item.title.text if item.title else 'No title',
                    'description': item.description.text if item.description else 'No description'
                })
        return jobs
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching jobs from Indeed: {str(e)}")
        return []

#------------------ Fetch Jobs with Progress -----------------#
def fetch_all_jobs(keyword, location, country):
    """Fetch jobs from all sources with progress tracking"""
    all_jobs = []
    
    with st.spinner('Fetching jobs...'):
        progress_bar = st.progress(0)
        
        # Fetch from Seek
        progress_bar.progress(25)
        seek_jobs = get_seek_jobs(keyword, location, country)
        all_jobs.extend(seek_jobs)
        
        # Fetch from Indeed
        progress_bar.progress(75)
        indeed_jobs = get_indeed_jobs(keyword, location, country)
        all_jobs.extend(indeed_jobs)
        
        progress_bar.progress(100)
    
    return all_jobs

# ----------------- Job Matching ----------------- #
def calculate_match(job_description, resume_skills):
    """Calculate match percentage between job and resume"""
    skills_str = " ".join(resume_skills)
    return fuzz.partial_ratio(skills_str.lower(), job_description.lower())

# ----------------- Main Application ----------------- #
def main():
    # Initialize NLP
    nlp = setup_nlp()
    
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
        try:
            # Parse resume
            resume_data = parse_resume("temp_resume.pdf", nlp)
            st.success(f"Skills Extracted: {', '.join(resume_data['skills'])}")
            
            # Fetch Jobs
            jobs = fetch_all_jobs(job_keyword, job_location, country)
            
            if not jobs:
                st.warning("No jobs found. This could be due to:")
                st.write("- Network connectivity issues")
                st.write("- Invalid location or keyword")
                st.write("- No job postings matching your criteria")
                return
            
            # Match Jobs
            matched_jobs = [
                job for job in jobs 
                if calculate_match(job['description'], resume_data['skills']) > 60
            ]
            
            # Display Results
            if matched_jobs:
                st.subheader(f"Found {len(matched_jobs)} Matching Jobs:")
                
                # Add source filters
                sources = list(set(job['source'] for job in matched_jobs))
                selected_sources = st.multiselect(
                    "Filter by source:",
                    sources,
                    default=sources
                )
                
                filtered_jobs = [
                    job for job in matched_jobs 
                    if job['source'] in selected_sources
                ]
                
                for job in filtered_jobs:
                    with st.expander(f"{job['title']} ({job['source']})"):
                        st.write(job['description'])
                        match_score = calculate_match(
                            job['description'], 
                            resume_data['skills']
                        )
                        st.progress(match_score / 100)
                        st.write(f"Match Score: {match_score}%")
                
                # Create downloadable results
                df = pd.DataFrame(filtered_jobs)
                st.download_button(
                    "Download Matched Jobs as CSV",
                    df.to_csv(index=False),
                    "matched_jobs.csv",
                    "text/csv"
                )
            else:
                st.warning("No suitable jobs found. Try adjusting your filters.")
            
            # Clean up
            os.remove("temp_resume.pdf")
        
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()