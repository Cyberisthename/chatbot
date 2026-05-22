import requests
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Any

class ResearchFetcher:
    """
    Fetches research papers from arXiv and PubMed APIs
    """
    
    ARXIV_BASE_URL = "http://export.arxiv.org/api/query?"
    PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def fetch_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch papers from arXiv
        """
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        response = requests.get(self.ARXIV_BASE_URL, params=params)
        if response.status_code != 200:
            print(f"Error fetching from arXiv: {response.status_code}")
            return []
        
        root = ET.fromstring(response.content)
        # arXiv uses Atom format
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            paper = {
                "source": "arXiv",
                "id": entry.find('atom:id', ns).text.split('/')[-1],
                "title": entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                "summary": entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
                "authors": [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                "published": entry.find('atom:published', ns).text,
                "link": entry.find('atom:link[@title="pdf"]', ns).attrib['href'] if entry.find('atom:link[@title="pdf"]', ns) is not None else entry.find('atom:link', ns).attrib['href']
            }
            papers.append(paper)
            
        return papers

    def fetch_pubmed(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch papers from PubMed using E-utilities
        """
        # 1. Search for IDs
        search_url = f"{self.PUBMED_BASE_URL}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        search_response = requests.get(search_url, params=search_params)
        if search_response.status_code != 200:
            print(f"Error searching PubMed: {search_response.status_code}")
            return []
        
        id_list = search_response.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
            
        # 2. Fetch details for these IDs
        fetch_url = f"{self.PUBMED_BASE_URL}esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        if fetch_response.status_code != 200:
            print(f"Error fetching PubMed summaries: {fetch_response.status_code}")
            return []
            
        results = fetch_response.json().get("result", {})
        papers = []
        for pmid in id_list:
            if pmid in results:
                entry = results[pmid]
                paper = {
                    "source": "PubMed",
                    "id": pmid,
                    "title": entry.get("title", ""),
                    "summary": entry.get("description", "No summary available"), # PubMed summary is often limited in esummary
                    "authors": [author.get("name", "") for author in entry.get("authors", [])],
                    "published": entry.get("pubdate", ""),
                    "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
                # For PubMed we might want to try and get the abstract if possible, but esummary doesn't provide it
                # For this prototype, the title and metadata might suffice for some adapters
                papers.append(paper)
                
        return papers

if __name__ == "__main__":
    fetcher = ResearchFetcher()
    print("Testing arXiv fetch (quant-ph)...")
    arxiv_papers = fetcher.fetch_arxiv("cat:quant-ph", max_results=2)
    for p in arxiv_papers:
        print(f"- {p['title']} ({p['id']})")
        
    print("\nTesting PubMed fetch (Quantum Biology)...")
    pubmed_papers = fetcher.fetch_pubmed("Quantum Biology", max_results=2)
    for p in pubmed_papers:
        print(f"- {p['title']} ({p['id']})")
