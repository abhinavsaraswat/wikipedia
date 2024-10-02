import streamlit as st
import requests
from transformers import pipeline
import torch

# Set page configuration
st.set_page_config(page_title="Wikipedia Search App", layout="wide")

# Title of the app
st.title("Wikipedia Search App")

# Check if MPS (Metal Performance Shaders) is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = 0  # MPS devices are assigned device IDs starting from 0
    st.write("Using MPS device")
elif torch.cuda.is_available():
    device = 0  # CUDA device
    st.write("Using CUDA device")
else:
    device = -1  # CPU
    st.write("Using CPU device")

# Initialize the zero-shot classification pipeline
@st.cache_resource(show_spinner=False)
def load_classifier():
    classifier = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli',
        device=device
    )
    return classifier

classifier = load_classifier()

# Define candidate labels (categories)
candidate_labels = ['Person', 'Organization', 'Location', 'Event', 'Product', 'Work of Art', 'Other']

# Input from the user
search_term = st.text_input("Enter a search term:", value="the magic roundabout")

if st.button("Search"):
    # Start rendering outputs immediately
    # We can use a placeholder for progress bar
    progress_bar = st.progress(0)
    total_steps = 5  # Number of articles
    step = 0

    # Use a container to hold the search results
    results_container = st.container()

    with st.spinner("Searching Wikipedia..."):
        # Step 1: Search Wikipedia and get top 5 results
        search_url = "https://en.wikipedia.org/w/api.php"

        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': search_term,
            'format': 'json',
            'srlimit': 5  # Limit to top 5 results
        }

        response = requests.get(search_url, params=search_params)
        data = response.json()

        # Step 2: Retrieve the titles of the top 5 results
        search_results = data['query']['search']
        titles = [result['title'] for result in search_results]

    # For each title, retrieve introduction and image, and process it
    for idx, title in enumerate(titles):
        with st.spinner(f"Processing article {idx+1}/{len(titles)}: {title}"):
            # Step 3: Retrieve content for the title
            content_url = "https://en.wikipedia.org/w/api.php"

            content_params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|pageimages',
                'exintro': True,       # Get only the introduction part
                'explaintext': True,   # Get plain text content for NLP processing
                'titles': title,
                'piprop': 'thumbnail',  # Get page thumbnail
                'pithumbsize': 200,     # Set thumbnail size
                'pilimit': 1,
            }

            content_response = requests.get(content_url, params=content_params)
            content_data = content_response.json()

            pages = content_data['query']['pages']
            page = next(iter(pages.values()))
            page_title = page.get('title', '')
            page_url = "https://en.wikipedia.org/wiki/" + page_title.replace(' ', '_')
            extract = page.get('extract', '')
            thumbnail = page.get('thumbnail', {}).get('source', '')

            # Perform zero-shot classification on the introduction text
            if extract:
                classification = classifier(extract, candidate_labels, multi_label=True)
                # Get labels and scores
                labels = classification['labels']
                scores = classification['scores']
                # Prepare categories with probabilities
                categories_with_probs = [(label, round(score, 4)) for label, score in zip(labels, scores)]
            else:
                categories_with_probs = [('Unknown', 0.0)]

            # Display the result for this article
            with results_container:
                st.markdown(f"### {page_title}")
                st.write(f"**URL:** [{page_url}]({page_url})")

                cols = st.columns([1, 3])

                with cols[0]:
                    if thumbnail:
                        st.image(thumbnail, use_column_width=True)
                    else:
                        st.write("No image available")

                with cols[1]:
                    st.write(extract)
                    st.write("**Categories with probabilities:**")
                    for label, score in categories_with_probs:
                        st.write(f"- {label}: {score}")

                st.markdown("---")

            # Update the progress bar
            step += 1
            progress_bar.progress(step / total_steps)

    # When done, remove the progress bar
    progress_bar.empty()