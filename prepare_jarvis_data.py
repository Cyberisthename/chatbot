
import requests
import json
import time

BOOKS = [
    {"title": "On the Origin of Species", "author": "Charles Darwin", "url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt", "year": 1859},
    {"title": "Relativity: The Special and General Theory", "author": "Albert Einstein", "url": "https://www.gutenberg.org/cache/epub/30155/pg30155.txt", "year": 1916},
    {"title": "Experiments in Plant-Hybridisation", "author": "Gregor Mendel", "url": "https://www.gutenberg.org/cache/epub/45634/pg45634.txt", "year": 1865},
    {"title": "The Atomic Theory", "author": "Adolphe Wurtz", "url": "https://www.gutenberg.org/cache/epub/41838/pg41838.txt", "year": 1880},
    {"title": "An Introduction to the Study of Experimental Medicine", "author": "Claude Bernard", "url": "https://www.gutenberg.org/cache/epub/60931/pg60931.txt", "year": 1865}
]

def download_books():
    print("🚀 Downloading historical scientific books from Project Gutenberg...")
    filtered_data = []
    
    for book in BOOKS:
        print(f"📥 Downloading {book['title']} ({book['year']})...")
        try:
            response = requests.get(book['url'])
            if response.status_code == 200:
                text = response.text
                # Clean up a bit (remove Gutenberg header/footer if possible, but keeping it for now is fine)
                filtered_data.append({
                    "text": text,
                    "title": book['title'],
                    "author": book['author'],
                    "year": book['year']
                })
                print(f"✅ Success: {len(text)} characters")
            else:
                print(f"❌ Failed to download {book['title']}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error downloading {book['title']}: {e}")
        time.sleep(1) # Be nice to Gutenberg
        
    with open("filtered_books.json", "w") as f:
        json.dump(filtered_data, f)
    
    print(f"🏁 Saved {len(filtered_data)} books to filtered_books.json")

if __name__ == "__main__":
    download_books()
