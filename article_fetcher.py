import requests
from bs4 import BeautifulSoup
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import re
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime
import time
import sys
import chardet
import ftfy
sys.path.append('..')
import app_config
from .dom_extractor import DOMExtractor
from .cache_utils import normalize_url, text_hash
from .content_extractor import apply_junk_filter, check_extraction_gates, calculate_cqs
# from .telemetry import ExtractionTelemetry  # Removed
from .selenium_helper import fetch_html_selenium, POOL
from .ensemble_extractor import EnsembleExtractor, ExtractionCandidate
from .html_normalizer import compute_hashes

def is_garbled_text(text: str, min_length: int = 100) -> bool:
    """Detect if text is garbled/obfuscated (like ROT encoding)."""
    if not text or len(text) < min_length:
        return False
    
    # Check for high density of special characters and low vowel ratio
    vowels = 'aeiouAEIOU'
    vowel_count = sum(1 for c in text[:500] if c in vowels)
    alnum_count = sum(1 for c in text[:500] if c.isalnum())
    
    if alnum_count == 0:
        return False
    
    vowel_ratio = vowel_count / alnum_count
    
    # Normal English has ~38% vowels, garbled text usually < 15%
    if vowel_ratio < 0.15:
        # Additional check for patterns like "E96 8=@32="
        weird_pattern_count = len(re.findall(r'[A-Z0-9]{2,}[@=:;]{1,}', text[:200]))
        if weird_pattern_count > 3:
            return True
    
    return False

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

try:
    from readability import Readability
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleFetcher:
    def __init__(self):
        self.session = self._create_session()
        self.max_workers = 10
        self.timeout = 10
        self.pre_filter_config = self._load_pre_filter_config()
        # No need for driver management - POOL handles it
        self.dom_extractor = DOMExtractor()  # Initialize DOM extractor
        self._content_cache = {}  # Cache for duplicate content detection
        # Telemetry removed
        self._html_extract_cache = {}  # Cache for HTML extraction results
        self.domain_rules = self._load_domain_rules()
        self.ensemble = None  # Lazy init after readability/trafilatura imports
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup_selenium()
        if self.ensemble:
            del self.ensemble
        
    def _create_session(self):
        session = requests.Session()
        # Simpler headers that match what worked in simple_test
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        return session
    
    def _load_pre_filter_config(self):
        config_path = app_config.PRE_FILTER_CONFIG_PATH
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "paywall_domains": [
                "ft.com", "mondaq.com", "mining-journal.com", 
                "reuters.com", "bloomberg.com", "3dadept.com", 
                "wsj.com", "barrons.com", "foreignpolicy.com"
            ],
            "excluded_patterns": [
                "youtube.com", "youtu.be", "/video/", "vimeo.com",
                "twitter.com", "x.com", "facebook.com", "linkedin.com",
                "instagram.com", "tiktok.com", "/podcast/", "podcasts."
            ]
        }
    
    def _load_domain_rules(self):
        """Load domain-specific extraction rules."""
        rules_path = Path(app_config.CONFIG_DIR) / 'domain_extraction_rules.json'
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _should_exclude(self, url):
        # Check paywall domains
        domain = urlparse(url).netloc.lower()
        if any(pd in domain for pd in self.pre_filter_config.get("paywall_domains", [])):
            return True, "paywall"
        
        # Check excluded patterns
        url_lower = url.lower()
        for pattern in self.pre_filter_config.get("excluded_patterns", []):
            if pattern in url_lower:
                return True, "excluded_pattern"
        
        return False, None
    
    def _extract_content(self, url, html):
        """Use ensemble extraction with caching."""
        # Initialize ensemble on first use
        if self.ensemble is None:
            self.ensemble = EnsembleExtractor(
                readability_func=self._extract_with_readability,
                trafilatura_func=self._extract_with_trafilatura,
                domain_rules=self.domain_rules
            )
        
        # Check HTML extraction cache
        url_hash, html_hash = compute_hashes(url, html)
        cache_key = f"{url_hash}_{html_hash}"
        
        if cache_key in self._html_extract_cache:
            cached = self._html_extract_cache[cache_key]
            if time.time() - cached.get('ts', 0) < 86400:  # 24h TTL
                logger.info(f"HTML extraction cache hit for {url[:50]}")
                return cached['content'], cached['method'], cached.get('metadata', {})
        
        # Run ensemble extraction
        best_candidate, stop_reason = self.ensemble.extract_best(html, url)
        
        # Handle AMP/print redirect
        if stop_reason == "amp_print_redirect" and isinstance(best_candidate, dict):
            logger.info(f"AMP/print redirect suggested: {best_candidate['redirect_url'][:50]}")
            # Return special marker for caller to handle
            return None, "redirect", best_candidate
        
        # Cache the result
        self._html_extract_cache[cache_key] = best_candidate.to_cache_dict()
        
        # Telemetry removed
        
        logger.info(f"Ensemble extraction: {best_candidate.method} "
                   f"(CQS: {best_candidate.cqs_score:.2f}, stop: {stop_reason})")
        
        return best_candidate.content, best_candidate.method, best_candidate.metadata
    
    def _extract_with_trafilatura(self, html, url):
        """Extract using trafilatura for ensemble."""
        if not HAS_TRAFILATURA:
            return "", {}
        
        try:
            content = trafilatura.extract(html, include_comments=False,
                                        include_tables=False, deduplicate=True)
            if content:
                return content, {}
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed: {e}")
        return "", {}
    
    def _extract_with_beautifulsoup(self, html):
        if not html:
            logger.warning("No HTML provided to BeautifulSoup")
            return ""
        logger.debug(f"BeautifulSoup processing {len(html)} chars of HTML")
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements - expanded denylist
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
            element.decompose()
        
        # Remove social/junk by class/id
        for selector in [
            '[aria-hidden="true"]', '.share', '.social', '.newsletter', '.breadcrumbs',
            '.comments', '.related', '.recommend', '.ads', '.ad', '.advertisement',
            '.sponsor', '.live-blog', '.photo-credit', '.byline', '.tag-list',
            '.subscribe', '.follow', '.print', '.email'
        ]:
            for el in soup.select(selector):
                el.decompose()
        
        # More comprehensive selectors
        content_selectors = [
            # Common article containers
            ('article', None),
            ('div', {'class': re.compile(r'entry-content|post-content|article-content|content-area|main-content')}),
            ('div', {'class': re.compile(r'story|post|entry|article')}),
            ('div', {'id': re.compile(r'content|main|article|post')}),
            ('main', None),
            ('section', {'class': re.compile(r'content|article|post')}),
            # WordPress common patterns
            ('div', {'class': 'entry'}),
            ('div', {'class': 'post-entry'}),
            # News site patterns
            ('div', {'class': re.compile(r'article-body|story-body|post-body')}),
            ('div', {'itemprop': 'articleBody'}),
        ]
        
        # Try each selector
        for tag, attrs in content_selectors:
            if attrs:
                elements = soup.find_all(tag, attrs)
            else:
                elements = soup.find_all(tag)
                
            for element in elements:
                # Get text and check if substantial
                text = element.get_text(separator='\n', strip=True)
                # Count paragraphs within element
                paras = element.find_all('p')
                # Lower threshold and better check
                if len(paras) >= 2 or len(text) > 300:
                    logger.debug(f"Found content in {tag} with {len(paras)} paragraphs, {len(text)} chars")
                    return self._clean_text(text)
        
        # Simpler paragraph extraction
        paragraphs = []
        all_p_tags = soup.find_all('p')
        
        for p in all_p_tags:
            text = p.get_text(strip=True)
            
            # Skip short or junk paragraphs
            if len(text) < 30:
                continue
                
            # Skip common footer/header text
            skip_terms = ['cookie', 'privacy policy', 'subscribe', 'newsletter', 'copyright', '©']
            if any(term in text.lower() for term in skip_terms):
                continue
            
            paragraphs.append(text)
        
        content = '\n'.join(paragraphs)
        return self._clean_text(content) if content else ""
    
    def _extract_with_readability(self, html, url):
        """Extract content using readability-lxml."""
        if not HAS_READABILITY or not html:
            return "", {}
        
        try:
            doc = Readability(html, url)
            result = doc.summary()
            
            if result:
                soup = BeautifulSoup(result, 'html.parser')
                content = soup.get_text(separator=' ', strip=True)
                return content, {'title': doc.title()}
            
        except Exception as e:
            logger.debug(f"Readability extraction failed: {e}")
        
        return "", {}
    
    def _try_selenium_fallback(self, url):
        """Try Selenium with decoupled timeout."""
        if not app_config.SELENIUM_SIMPLE_MODE:
            # Legacy ensemble mode if needed
            return None
            
        try:
            # Fetch with independent timeout
            html, meta = fetch_html_selenium(url)
            
            # Extract content using our existing method
            content, extraction_method, dom_metadata = self._extract_content(url, html)
            
            # Apply junk filter
            if content:
                content = apply_junk_filter(content)
            
            # Check extraction gates
            gate_passed, gate_reason = check_extraction_gates(content or '')
            
            # Calculate CQS
            title = dom_metadata.get('title', '') if dom_metadata else ''
            cqs_metrics = calculate_cqs(content or '', title)
            
            if content and gate_passed:
                # Check content cache
                content_key = text_hash(content)
                if content_key in self._content_cache:
                    cached_result = self._content_cache[content_key]
                    logger.info(f"Duplicate content detected (Selenium) for {url[:50]}...")
                    return cached_result
                
                logger.info(f"Selenium succeeded: {len(content)} chars from {url[:50]}... (CQS: {cqs_metrics['score']})")
                result = {
                    'content': content,
                    'status': 'success',
                    'length': len(content),
                    'method': extraction_method,
                    'fetch_method': 'selenium',
                    'gate_passed': True,
                    'gate_reason': gate_reason,
                    'cqs_score': cqs_metrics['score'],
                    'cqs_metrics': cqs_metrics,
                    **meta  # Include selenium metrics
                }
                self._content_cache[content_key] = result
                
                # Telemetry removed
                
                return result
            else:
                logger.warning(f"Selenium content failed gates: {gate_reason}")
                result = {
                    'content': content or '',
                    'status': 'gate_failed',
                    'error': f'Gates failed: {gate_reason}',
                    'method': extraction_method if content else 'none',
                    'fetch_method': 'selenium',
                    'gate_passed': gate_passed,
                    'gate_reason': gate_reason,
                    'cqs_score': cqs_metrics['score'],
                    'cqs_metrics': cqs_metrics,
                    **meta  # Include selenium metrics
                }
                
                # Telemetry removed
                
                return result
                
        except Exception as e:
            if "captcha_or_403" in str(e):
                logger.warning(f"Selenium blocked by captcha/403 on {url[:50]}...")
            else:
                logger.error(f"Selenium fallback failed on {url[:50]}...: {e}")
            return None
    
    def _clean_text(self, text):
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _extract_article_core(self, content, title):
        """Extract core article content, skipping headers/ads"""
        paragraphs = content.split('\n')
        content_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:
                continue
            
            # Skip common junk patterns
            junk_patterns = [
                'cookie', 'subscribe', 'advertisement', 'follow us',
                'share this', 'print', 'email', 'published:', 'by:',
                'copyright', '©', 'all rights reserved', 'terms of service'
            ]
            if any(pattern in para.lower() for pattern in junk_patterns):
                continue
            
            content_paragraphs.append(para)
            
            # Get 2-3 good paragraphs (usually 500-800 chars)
            if len(' '.join(content_paragraphs)) > 600:
                break
        
        # Fallback: look for title keywords in content
        if not content_paragraphs and title:
            title_words = set(title.lower().split())
            for para in paragraphs:
                if len(para) > 50:
                    para_words = set(para.lower().split())
                    if len(title_words & para_words) >= 2:
                        content_paragraphs.append(para)
                        break
        
        return ' '.join(content_paragraphs[:3])
    
    def fetch_article(self, url):
        original_url = url  # Keep original for result key
        try:
            # Handle Google News redirect URLs
            if 'google.com/url' in url:
                import urllib.parse
                parsed = urllib.parse.urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                if 'url' in params:
                    actual_url = params['url'][0]
                    logger.info(f"Google redirect: {url[:50]}... -> {actual_url}")
                    url = actual_url
            
            # Normalize URL for cache lookup
            normalized_url = normalize_url(url)
            
            # Check if should exclude
            should_exclude, reason = self._should_exclude(url)
            if should_exclude:
                logger.info(f"Excluded {url[:50]}... - Reason: {reason}")
                return original_url, {
                    'content': '',
                    'status': reason,
                    'error': f'Excluded: {reason}'
                }
            
            # Fetch HTML
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Robust encoding detection
            raw_content = response.content
            # Try charset detection on raw bytes
            detected = chardet.detect(raw_content)
            encoding = detected['encoding'] if detected['confidence'] > 0.7 else None
            
            # Fallback order: detected -> meta tag -> headers -> utf-8
            if not encoding:
                # Quick check for meta charset in first 1024 bytes
                head = raw_content[:1024].decode('utf-8', errors='ignore').lower()
                meta_match = re.search(r'charset=["\']?([\w-]+)', head)
                if meta_match:
                    encoding = meta_match.group(1)
            
            html_text = raw_content.decode(encoding or response.encoding or 'utf-8', errors='replace')
            # Fix any remaining encoding issues
            html_text = ftfy.fix_text(html_text)
            
            # Extract content
            content, extraction_method, dom_metadata = self._extract_content(url, html_text)
            
            # Check for garbled/obfuscated content
            if content and is_garbled_text(content):
                logger.warning(f"Garbled content detected for {url[:50]}...")
                return original_url, {
                    'content': '',
                    'status': 'garbled',
                    'error': 'Unable to extract content - site uses anti-scraping measures'
                }
            
            # Apply junk filter
            if content:
                content = apply_junk_filter(content)
            
            # Check extraction gates
            gate_passed, gate_reason = check_extraction_gates(content or '')
            
            # Calculate CQS
            title = dom_metadata.get('title', '') if dom_metadata else ''
            cqs_metrics = calculate_cqs(content or '', title)
            
            if content and gate_passed:
                # Check if we've seen this exact content before
                content_key = text_hash(content)
                if content_key in self._content_cache:
                    cached_result = self._content_cache[content_key]
                    logger.info(f"Duplicate content detected for {url[:50]}...")
                    return original_url, cached_result
                
                logger.info(f"Successfully extracted {len(content)} chars from {url[:50]}... (CQS: {cqs_metrics['score']})")
                result = {
                    'content': content,
                    'status': 'success',
                    'length': len(content),
                    'method': extraction_method,
                    'fetch_method': 'requests',
                    'gate_passed': True,
                    'gate_reason': gate_reason,
                    'cqs_score': cqs_metrics['score'],
                    'cqs_metrics': cqs_metrics
                }
                
                # Cache the result
                self._content_cache[content_key] = result
                
                # Telemetry removed
                
                # Add DOM metadata if available
                if dom_metadata:
                    result['dom_metadata'] = dom_metadata
                    # Add key fields to top level for easy access
                    if dom_metadata.get('title'):
                        result['extracted_title'] = dom_metadata['title']
                    if dom_metadata.get('authors'):
                        result['extracted_authors'] = dom_metadata['authors']
                    if dom_metadata.get('published_at'):
                        result['extracted_date'] = dom_metadata['published_at']
                    if dom_metadata.get('confidence'):
                        result['extraction_confidence'] = dom_metadata['confidence']
                
                return original_url, result
            else:
                logger.warning(f"Content failed gates for {url[:50]}... Reason: {gate_reason}")
                # Return with gate failure info
                result = {
                    'content': content or '',
                    'status': 'gate_failed' if not gate_passed else ('partial' if content else 'failed'),
                    'error': f'Gates failed: {gate_reason}' if not gate_passed else f'Minimal content ({len(content) if content else 0} chars)',
                    'method': extraction_method if content else 'none',
                    'fetch_method': 'requests',
                    'gate_passed': gate_passed,
                    'gate_reason': gate_reason,
                    'cqs_score': cqs_metrics['score'],
                    'cqs_metrics': cqs_metrics
                }
                
                # Telemetry removed
                
                return original_url, result
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url[:50]}...")
            return original_url, {'content': '', 'status': 'timeout', 'error': 'Timeout'}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                domain = urlparse(url).netloc.lower()
                logger.warning(f"403 Forbidden for {domain} - Trying Selenium fallback")
                
                # Try Selenium as fallback for 403 errors
                selenium_result = self._try_selenium_fallback(url)
                if selenium_result:
                    return original_url, selenium_result
                
                # If Selenium also failed, return blocked status
                return original_url, {
                    'content': '', 
                    'status': 'blocked', 
                    'error': f'403 Forbidden - {domain} blocks all methods',
                    'method': 'none',
                    'fetch_method': 'selenium_failed'
                }
            else:
                logger.error(f"HTTP error for {url[:50]}...: {str(e)}")
                return original_url, {'content': '', 'status': 'error', 'error': str(e)}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url[:50]}...: {str(e)}")
            return original_url, {'content': '', 'status': 'error', 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error for {url[:50]}...: {str(e)}")
            return original_url, {'content': '', 'status': 'error', 'error': str(e)}
    
    def fetch_articles(self, df, progress_callback=None):
        urls = df['Link'].tolist()
        results = {}
        total = len(urls)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map original URLs to futures
            future_to_original_url = {}
            for url in urls:
                future = executor.submit(self.fetch_article, url)
                future_to_original_url[future] = url
            
            for future in as_completed(future_to_original_url):
                original_url = future_to_original_url[future]
                actual_url, result = future.result()
                # Store result with ORIGINAL URL as key (the Google redirect URL)
                results[original_url] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, f"Fetching Articles ({completed}/{total})")
        
        # Clean up Selenium if it was used
        self._cleanup_selenium()
        
        return results
    
    def _cleanup_selenium(self):
        """Clean up Selenium pool."""
        try:
            POOL.quit()
            logger.info("Cleaned up Selenium pool")
        except Exception as e:
            logger.error(f"Error cleaning Selenium pool: {e}")
    
    def force_quit_driver(self):
        """Force quit Selenium pool"""
        try:
            POOL.quit()
            # Also check ensemble extractor
            if hasattr(self, 'ensemble') and self.ensemble and hasattr(self.ensemble, 'driver'):
                self.ensemble.driver.quit()
        except Exception:
            pass