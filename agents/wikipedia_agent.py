# agents/wikipedia_agent.py

import wikipedia

def fetch_wikipedia_summary(topic):
    """
    Fetch a summary for a given topic from Wikipedia.
    
    Args:
        topic (str): The topic to search on Wikipedia.
    
    Returns:
        str: Summary text from Wikipedia.
    """
    try:
        # Get the summary from Wikipedia for the given topic
        summary = wikipedia.summary(topic, sentences=3)  # Limiting to 3 sentences for brevity
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found for {topic}. Please be more specific."
    except wikipedia.exceptions.HTTPTimeoutError:
        return "The request timed out. Please try again later."
    except wikipedia.exceptions.RequestError:
        return "There was an error while trying to fetch the data."
