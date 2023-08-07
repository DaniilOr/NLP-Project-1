# import the necessary modules
import argparse
import trafilatura
import trafilatura.spider as ts
import json
from tqdm import tqdm

def extract_webpage(webpage_url):
    """
    Extracts metadata and raw text from a given webpage URL.

    Args:
        webpage_url (str): The URL of the webpage to extract.

    Returns:
        dict: A dictionary object containing raw html and processed html.
    """
    downloaded_raw_html = trafilatura.fetch_url(webpage_url)
    from trafilatura.settings import use_config
    newconfig = use_config()
    newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

    # discard potential comment and change the output to JSON
    result = trafilatura.extract(
        downloaded_raw_html,
        output_format="json",
        url=webpage_url,
        include_comments=True,
        include_images=True,
        include_links=True,
        include_tables=True,
        config=newconfig
    )

    result = json.loads(result)
    print(result)
    return {"raw_html": downloaded_raw_html, "processed_data": result}


def extract_website(website_base_url):
    """
    Extracts webpages from a given website and applies the `extract_webpage` function to each webpage.

    Args:
        website_base_url (str): The base URL of the website to extract.

    Returns:
        dict: A dictionary containing the website base URL as key and the value as a dictionary of webpages data as returned by the `extract_webpage` function.
    """
    webpages = ts.focused_crawler(website_base_url, max_seen_urls=10,
                                  max_known_urls=50)

    all_webpages = set()
    for webpage_set in webpages:
        all_webpages = all_webpages | webpage_set

    data = dict()
    for webpage_url in tqdm(all_webpages):
        try:
            data[webpage_url] = extract_webpage(webpage_url)
        except Exception as e:
            print(f"An exception occurred: {str(e)}")
            pass

    return {website_base_url: data} 


if __name__ == "__main__":
    # Parse the command-line arguments
    # parser = argparse.ArgumentParser(description="Webpage extraction script")
    # parser.add_argument("website_base_url", type=str, help="Base URL of the website to extract")
    # args = parser.parse_args()

    # Extract the website
    data = extract_website(args.website_base_url)

    # Save the data to a JSON file
    # with open('data.json', 'w') as f:
    #     json.dump(data, f, indent=2)
