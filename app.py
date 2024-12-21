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
    """Fetch jobs from Seek with debugging"""
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
            st.warning(f"No Seek domain found for {country}")
            return []
        
        # URL encode the keyword and location
        keyword = requests.utils.quote(keyword)
        location = requests.utils.quote(location)
        base_url = f"https://www.{domain}/jobs?keywords={keyword}&where={location}"
        
        # Add user agent to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        st.info(f"Fetching from Seek: {base_url}")
        response = requests.get(base_url, headers=headers, verify=True, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            jobs = []
            
            # Debug info
            st.write(f"Seek Response Length: {len(response.content)}")
            
            # Look for job listings with different possible class names
            job_elements = (
                soup.find_all('article', class_='_1wkzzau0') or  # Try modern class
                soup.find_all('article', class_='job-card') or    # Try alternative class
                soup.find_all('div', class_='job-card') or       # Try div instead
                []
            )
            
            st.write(f"Found {len(job_elements)} job elements on Seek")
            
            for job in job_elements[:10]:  # Limit to first 10 jobs
                try:
                    title = (
                        job.find('h3', class_='job-title').text.strip() if job.find('h3', class_='job-title') else
                        job.find('a', class_='job-title').text.strip() if job.find('a', class_='job-title') else
                        job.find('span', class_='title').text.strip() if job.find('span', class_='title') else
                        'No title found'
                    )
                    
                    description = (
                        job.find('span', class_='job-description').text.strip() if job.find('span', class_='job-description') else
                        job.find('div', class_='job-description').text.strip() if job.find('div', class_='job-description') else
                        job.find('div', class_='description').text.strip() if job.find('div', class_='description') else
                        'No description found'
                    )
                    
                    if title != 'No title found' or description != 'No description found':
                        jobs.append({
                            'source': 'Seek',
                            'title': title,
                            'description': description
                        })
                except Exception as e:
                    st.warning(f"Error parsing job listing: {str(e)}")
                    continue
            
            return jobs
        else:
            st.warning(f"Seek returned status code: {response.status_code}")
            return []
            
    except Exception as e:
        st.warning(f"Error fetching jobs from Seek: {str(e)}")
        return []

def get_indeed_jobs(keyword, location, country):
    """Fetch jobs from Indeed with debugging"""
    try:
        country_domains = {
            "Australia": "au",
            "United States": "com",
            "United Kingdom": "co.uk",
            "Canada": "ca",
            "India": "co.in",
            "Germany": "de",
            "France": "fr",
            "Singapore": "sg",
            "United Arab Emirates": "ae",
            "Japan": "jp",
        }
        domain = country_domains.get(country)
        if not domain:
            st.warning(f"No Indeed domain found for {country}")
            return []
        
        # URL encode the keyword and location
        keyword = requests.utils.quote(keyword)
        location = requests.utils.quote(location)
        
        # Try both RSS and regular HTML endpoints
        urls = [
            f"https://{domain}.indeed.com/jobs?q={keyword}&l={location}",
            f"https://{domain}.indeed.com/rss?q={keyword}&l={location}"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        jobs = []
        for url in urls:
            st.info(f"Fetching from Indeed: {url}")
            response = requests.get(url, headers=headers, verify=True, timeout=10)
            
            if response.status_code == 200:
                st.write(f"Indeed Response Length: {len(response.content)}")
                
                if 'rss' in url:
                    # Parse RSS feed
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')
                    st.write(f"Found {len(items)} jobs in RSS feed")
                    
                    for item in items[:10]:  # Limit to first 10 jobs
                        jobs.append({
                            'source': 'Indeed',
                            'title': item.title.text if item.title else 'No title',
                            'description': item.description.text if item.description else 'No description'
                        })
                else:
                    # Parse HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    job_cards = (
                        soup.find_all('div', class_='job_seen_beacon') or
                        soup.find_all('div', class_='jobsearch-SerpJobCard') or
                        soup.find_all('div', class_='job-card') or
                        []
                    )
                    
                    st.write(f"Found {len(job_cards)} job cards in HTML")
                    
                    for card in job_cards[:10]:  # Limit to first 10 jobs
                        try:
                            title = (
                                card.find('h2', class_='jobTitle').text.strip() if card.find('h2', class_='jobTitle') else
                                card.find('a', class_='jobtitle').text.strip() if card.find('a', class_='jobtitle') else
                                'No title found'
                            )
                            
                            description = (
                                card.find('div', class_='job-snippet').text.strip() if card.find('div', class_='job-snippet') else
                                card.find('div', class_='summary').text.strip() if card.find('div', class_='summary') else
                                'No description found'
                            )
                            
                            if title != 'No title found' or description != 'No description found':
                                jobs.append({
                                    'source': 'Indeed',
                                    'title': title,
                                    'description': description
                                })
                        except Exception as e:
                            st.warning(f"Error parsing Indeed job card: {str(e)}")
                            continue
            
            else:
                st.warning(f"Indeed returned status code: {response.status_code}")
        
        return jobs
        
    except Exception as e:
        st.warning(f"Error fetching jobs from Indeed: {str(e)}")
        return []

# Update the fetch_all_jobs function to show more debugging info
def fetch_all_jobs(keyword, location, country):
    """Fetch jobs from all sources with detailed progress tracking"""
    all_jobs = []
    
    with st.spinner('Fetching jobs...'):
        progress_bar = st.progress(0)
        
        # Fetch from Seek
        st.write("Attempting to fetch jobs from Seek...")
        progress_bar.progress(25)
        seek_jobs = get_seek_jobs(keyword, location, country)
        st.write(f"Found {len(seek_jobs)} jobs from Seek")
        all_jobs.extend(seek_jobs)
        
        # Fetch from Indeed
        st.write("Attempting to fetch jobs from Indeed...")
        progress_bar.progress(75)
        indeed_jobs = get_indeed_jobs(keyword, location, country)
        st.write(f"Found {len(indeed_jobs)} jobs from Indeed")
        all_jobs.extend(indeed_jobs)
        
        progress_bar.progress(100)
        st.write(f"Total jobs found: {len(all_jobs)}")
    
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
        # Save the uploaded file as temp_resume.pdf
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
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