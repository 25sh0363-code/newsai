"""
Web scraping utilities for news article extraction.
"""

import requests
from typing import Optional, Dict
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Try to import newspaper3k, fallback to pure BeautifulSoup if it fails
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

class ArticleScraper:
    def __init__(self):
        """Initialize the article scraper."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_from_url(self, url: str) -> Dict[str, str]:
        """Extract article content from URL."""
        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                return {
                    'title': article.title,
                    'text': article.text,
                    'authors': article.authors,
                    'publish_date': article.publish_date,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                # Fallback to BeautifulSoup if newspaper3k fails
                return self._extract_with_beautifulsoup(url)
        else:
            # Use BeautifulSoup directly if newspaper3k is not available
            return self._extract_with_beautifulsoup(url)

    def _extract_with_beautifulsoup(self, url: str) -> Dict[str, str]:
        """Fallback extraction method using BeautifulSoup."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Extract title
            title = soup.find('h1')
            title = title.text.strip() if title else ''
            
            # Extract main content
            article_text = ''
            
            # Try common article content selectors
            content_selectors = [
                'article',
                '[role="main"]',
                '.article-content',
                '.post-content',
                '.entry-content',
                '#main-content'
            ]
            
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    article_text = ' '.join([p.text.strip() for p in content.find_all('p')])
                    break
            
            # If no content found with selectors, get all paragraphs
            if not article_text:
                article_text = ' '.join([p.text.strip() for p in soup.find_all('p')])
            
            return {
                'title': title,
                'text': article_text,
                'authors': [],
                'publish_date': None,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'title': '',
                'text': '',
                'authors': [],
                'publish_date': None,
                'success': False,
                'error': str(e)
            }

    def validate_url(self, url: str) -> bool:
        """Validate if the URL is properly formatted."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False