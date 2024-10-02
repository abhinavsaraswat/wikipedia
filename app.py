import streamlit as st
import requests
from transformers import pipeline

# Set page configuration
st.set_page_config(page_title="Wikipedia Search App", layout="wide")

# Title of the app
st.title("Wikipedia Search App")

# Updated introduction
st.markdown("""
This app allows you to search Wikipedia for a given term and displays the top 5 search results. For each result, it shows the **introduction text**, an **image** (if available), and the **categories** of the article. The categories are predicted using a zero-shot classification model.

The app uses the **Hugging Face Transformers** library and a smaller version of the **BART** model for zero-shot classification to improve performance. It also uses the **Wikipedia API** to retrieve search results and article content. The categories include: **Person**, **Organization**, **Location**, **Event**, **Product**, **Work of Art**, and **Other**.

To get started, enter a search term in the text box below and click the **'Search'** button.
""")

# Define candidate labels (categories)
candidate_labels = ['Person', 'Organization', 'Location', 'Event', 'Product', 'Work of Art', 'Other']

# Input from the user
search_term = st.text_input("Enter a search term:", value="the magic roundabout")

if st.button("Search"):
    # Step 1: Search Wikipedia and get top 5 results
    with st.spinner("Searching Wikipedia..."):
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

        # Retrieve the titles of the top 5 results
        search_results = data['query']['search']
        titles = [result['title'] for result in search_results]

    # Step 2: Display preliminary information and placeholders
    st.write("## Search Results")
    article_placeholders = []
    for title in titles:
        # Create a placeholder for the entire article
        article_placeholder = st.empty()
        article_placeholders.append(article_placeholder)
        # Display initial loading message
        article_placeholder.markdown(f"### {title}\nLoading article content and categories...")

    # Step 3: Load the classifier once (if not already loaded)
    if 'classifier' not in st.session_state:
        with st.spinner("Loading classification model... This may take a while."):
            # Initialize the zero-shot classification pipeline with a smaller model
            classifier = pipeline(
                'zero-shot-classification',
                model='valhalla/distilbart-mnli-12-1',  # Smaller model for faster performance
                device=-1  # Force CPU usage; adjust if GPU is available
            )
        st.session_state['classifier'] = classifier
    else:
        classifier = st.session_state['classifier']

    # Step 4: Process each article sequentially
    for idx, (title, article_placeholder) in enumerate(zip(titles, article_placeholders)):
        with st.spinner(f"Processing article {idx + 1}/{len(titles)}: {title}"):
            # Retrieve content for the title
            content_url = "https://en.wikipedia.org/w/api.php"

            content_params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|pageimages',
                'exintro': True,       # Get only the introduction part
                'explaintext': True,   # Get plain text content
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
            extract = page.get('extract', 'No introduction available.')
            thumbnail = page.get('thumbnail', {}).get('source', '')

            # Limit the introduction text to improve processing speed
            max_length = 512  # Adjust as needed
            extract = extract[:max_length]

            # Update the article placeholder with article content
            with article_placeholder.container():
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
                    # Placeholder for categories
                    category_placeholder = st.empty()
                    category_placeholder.write("Categorizing...")

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

            # Update the category placeholder with categories
            category_text = "**Categories with probabilities:**\n"
            for label, score in categories_with_probs:
                category_text += f"- {label}: {score}\n"
            category_placeholder.markdown(category_text)