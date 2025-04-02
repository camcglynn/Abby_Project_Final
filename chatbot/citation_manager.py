import logging
from typing import List, Dict, Any, Optional
import re
import urllib.parse

logger = logging.getLogger(__name__)


class Citation:
    """Represents a citation source for information."""

    def __init__(self,
                 source: str,
                 url: Optional[str] = None,
                 title: Optional[str] = None,
                 authors: Optional[List[str]] = None,
                 publication_date: Optional[str] = None,
                 accessed_date: Optional[str] = None):
        self.source = source
        self.url = url
        self.title = title
        self.authors = authors or []
        self.publication_date = publication_date
        self.accessed_date = accessed_date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "accessed_date": self.accessed_date
        }

    def to_html(self) -> str:
        html = f'<div class="citation">'
        if self.authors:
            if len(self.authors) == 1:
                html += f"{self.authors[0]}. "
            elif len(self.authors) == 2:
                html += f"{self.authors[0]} & {self.authors[1]}. "
            else:
                html += f"{self.authors[0]} et al. "
        if self.publication_date:
            html += f"({self.publication_date}). "
        if self.title:
            if self.url:
                html += f'<a href="{self.url}" target="_blank" rel="noopener noreferrer">{self.title}</a>. '
            else:
                html += f"{self.title}. "
        html += f'<span class="citation-source">{self.source}</span>'
        if self.accessed_date:
            html += f'. Accessed on {self.accessed_date}'
        html += '</div>'
        return html

    def to_markdown(self) -> str:
        md = ""
        if self.authors:
            if len(self.authors) == 1:
                md += f"{self.authors[0]}. "
            elif len(self.authors) == 2:
                md += f"{self.authors[0]} & {self.authors[1]}. "
            else:
                md += f"{self.authors[0]} et al. "
        if self.publication_date:
            md += f"({self.publication_date}). "
        if self.title:
            if self.url:
                md += f"[{self.title}]({self.url}). "
            else:
                md += f"{self.title}. "
        md += f"*{self.source}*"
        if self.accessed_date:
            md += f". Accessed on {self.accessed_date}"
        return md


class CitationManager:
    """
    Manages citations for chatbot responses.
    Sources will only be added from RAG chunks with valid URLs.
    """

    def __init__(self):
        logger.info("Initializing Citation Manager")
        self.sources = {}
        self.default_sources = []
        
    def add_custom_sources_from_rag(self, rag_chunks: List[Dict[str, Any]], filter_domains: bool = True) -> None:
        """
        Add custom sources from RAG chunks to the sources dictionary.
        Only includes real, valid URLs and skips programmatic/default sources.
        Deduplicates citations by URL to ensure each unique URL has only one citation.
        
        Args:
            rag_chunks: List of dictionaries containing RAG results with fields:
                        Question, Answer, Link, Category
            filter_domains: If True, only include links from approved domains
        """
        # List of approved domains for citations (in order of priority)
        approved_domains = [
            "plannedparenthood.org",  # Planned Parenthood is highest priority
            "acog.org",
            "cdc.gov",
            "nih.gov",
            "who.int",
            "mayoclinic.org",
            "healthline.com",
            "womenshealth.gov"
        ]
        
        # Patterns for non-existent or default URLs
        invalid_url_patterns = [
            "example.com",
            "example.org",
            "test.com",
            "localhost",
            "127.0.0.1",
            "placeholder"
        ]
        
        # Track existing URLs to avoid duplicates
        existing_links = {citation.url: source_id for source_id, citation in self.sources.items()}
        
        # Process and filter chunks
        filtered_chunks = []
        
        for chunk in rag_chunks:
            # Skip if Link is missing or empty
            if "Link" not in chunk or not chunk["Link"]:
                continue
            
            # Handle non-string Link values (convert to string or skip)
            link = chunk["Link"]
            if not isinstance(link, str):
                try:
                    # Try to convert to string if possible
                    link = str(link)
                    # Skip if it's an empty string after conversion
                    if not link or link.lower() == "nan":
                        logger.info(f"Skipping citation with non-valid Link after conversion: {link}")
                        continue
                    # Update the Link in the chunk
                    chunk["Link"] = link
                except:
                    logger.info(f"Skipping citation with non-string Link that couldn't be converted: {link}")
                    continue
                
            # Skip invalid or default URLs
            if any(pattern in link.lower() for pattern in invalid_url_patterns):
                logger.info(f"Skipping citation with invalid or default URL: {link}")
                continue
                
            # Extract domain from the URL
            try:
                parsed_url = urllib.parse.urlparse(link)
                
                # Skip URLs without a proper domain or scheme
                if not parsed_url.netloc or not parsed_url.scheme:
                    logger.info(f"Skipping citation with malformed URL: {link}")
                    continue
                    
                domain = parsed_url.netloc
                # Remove www. prefix if present
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception as e:
                logger.info(f"Skipping citation with invalid URL format: {link} - {str(e)}")
                continue
            
            # Check domain against approved list if filtering is enabled
            if filter_domains and not any(approved_domain in domain for approved_domain in approved_domains):
                logger.info(f"Skipping citation from non-approved domain: {domain}")
                continue
            
            # URL passed all checks, add to filtered chunks
            filtered_chunks.append((chunk, domain, link))
        
        # Process the filtered chunks
        for chunk, domain, link in filtered_chunks:
            question = chunk.get("Question", "")
            
            # If this URL already exists in our sources, skip creating a new Citation
            if link in existing_links:
                logger.debug(f"Reusing existing citation for URL: {link}")
                continue
                
            # Create a unique key for this citation
            key = f"custom_{len(existing_links)}_{domain.replace('.', '_')}"
            
            # Set appropriate author list and source name based on domain
            if "plannedparenthood.org" in domain:
                authors = ["Planned Parenthood Federation of America"]
                source_name = "Planned Parenthood"
            elif "acog.org" in domain:
                authors = ["American College of Obstetricians and Gynecologists"]
                source_name = "ACOG"
            elif "cdc.gov" in domain:
                authors = ["Centers for Disease Control and Prevention"]
                source_name = "CDC"
            elif "nih.gov" in domain:
                authors = ["National Institutes of Health"]
                source_name = "NIH"
            elif "who.int" in domain:
                authors = ["World Health Organization"]
                source_name = "WHO"
            elif "mayoclinic.org" in domain:
                authors = ["Mayo Clinic Staff"]
                source_name = "Mayo Clinic"
            elif "healthline.com" in domain:
                authors = ["Healthline Editorial Team"]
                source_name = "Healthline"
            elif "womenshealth.gov" in domain:
                authors = ["Office on Women's Health"]
                source_name = "Women's Health.gov"
            else:
                authors = []
                source_name = domain
            
            # Create Citation object with the exact link from RAG
            self.sources[key] = Citation(
                source=source_name,
                url=link,  # Use the exact link from RAG
                title=question,
                authors=authors
            )
            
            # Add to existing links tracking
            existing_links[link] = key

    def add_citation_to_text(self, text: str, source_id: str) -> str:
        """
        Append a footnote citation marker (e.g. [^source_id]) to the text
        if the source_id exists and the text isn't conversational.
        """
        if source_id in self.sources and not self._is_conversational(text):
            # Only add citation if the text is substantial (not conversational)
            # Using consistent threshold of 10 words across all methods
            if len(text.split()) > 10:
                # Create footnote marker
                marker = f"[^{source_id}]"
                
                # Add the marker if not already present
                if marker not in text:
                    # If text already ends with a period, add marker after it
                    if text.rstrip().endswith(('.', '!', '?')):
                        return f"{text} {marker}"
                    # Otherwise, ensure the text has proper punctuation and then add marker
                    else:
                        return f"{text}. {marker}"
        return text

    def _is_conversational(self, text: str) -> bool:
        """
        Determine if text is a conversational response (greeting, question, etc.)
        rather than substantial information.
        """
        text_lower = text.lower()
        conversation_indicators = [
            "how can i help", "i'm doing well", "how are you",
            "thanks for asking", "could you let me know", "feel free to ask",
            "please let me know", "i'd like to provide",
            "i understand this can be", "feeling", "emotions", "feelings",
            "it's okay to", "many people experience", "your feelings are",
            "be gentle with yourself", "emotional", "i'd like to provide",
            "i understand this is", "guilt", "regret", "shame", "fear"
        ]

        # Check for question marks and short sentences
        # Using consistent threshold of 10 words across all methods
        is_question = "?" in text and len(text.split()) < 10

        # Check for conversation indicators
        has_indicator = any(indicator in text_lower
                            for indicator in conversation_indicators)
            
        # Check if text is primarily emotional support rather than factual information
        emotional_support_indicators = [
            "guilt", "regret", "shame", "fear", "sadness", 
            "your feelings are valid", "it's okay to feel", 
            "many people experience", "be gentle with yourself",
            "your worth isn't defined by", "you deserve compassion"
        ]
        
        is_emotional_support = any(indicator in text_lower 
                                for indicator in emotional_support_indicators)

        # Determine if it's conversational based on these factors
        # Lowered threshold to be consistent with conversation_manager.py
        return is_question or has_indicator or is_emotional_support or len(text.split()) < 10

    def extract_citations_from_text(self, text: str) -> List[Citation]:
        """
        Parse and return a list of Citation objects from footnote markers in the text.
        """
        citations = []

        # Skip if text is too short or conversational
        # Using consistent threshold of 10 words across all methods
        if self._is_conversational(text) or len(text.split()) < 10:
            return []

        # Look for footnote markers like [^source_id]
        footnote_pattern = r'\[\^([\w_]+)\]'
        footnote_matches = re.findall(footnote_pattern, text)

        for source_id in footnote_matches:
            if source_id in self.sources:
                citations.append(self.sources[source_id])

        return citations

    def format_response_with_citations(self, text: str, format_type: str = "html") -> Dict[str, Any]:
        """
        Format a response with extracted citations.

        This method:
        1. Extracts citations from text based on footnote markers
        2. Reformats the text to use numbered citations instead of footnotes
        3. Returns a dictionary with the cleaned text, formatted citations, and citation objects

        Args:
            text (str): Response text with footnote citations
            format_type (str): "html" or "markdown"

        Returns:
            Dict with keys:
            - text: Cleaned text with properly formatted citations
            - citations: List of formatted citation strings
            - citation_objects: List of citation objects as dictionaries
        """
        # Extract citations from text
        pattern = r'\[\^([^\]]+)\]'
        footnotes = re.findall(pattern, text)
        
        # First, clean any malformed citation brackets like [.
        clean_text = re.sub(r'\[\.\s*', '', text)
        clean_text = re.sub(r'\s?\[\.?\]', '', clean_text)
        
        # Clean the text by replacing all footnotes with numbered citations
        seen_source_ids = []
        source_id_to_number = {}
        
        # Keep track of which source_ids we've seen
        for source_id in footnotes:
            if source_id in self.sources and source_id not in seen_source_ids:
                seen_source_ids.append(source_id)
                source_id_to_number[source_id] = len(seen_source_ids)
        
        # Replace footnotes with numbered citations and ensure they're in the text
        for source_id in footnotes:
            if source_id in source_id_to_number:
                number = source_id_to_number[source_id]
                clean_text = clean_text.replace(f"[^{source_id}]", f"[{number}]")
        
        # Create formatted citations and citation objects
        formatted_citations = []
        citation_objects = []
        
        for source_id in seen_source_ids:
            citation = self.sources[source_id]
            number = source_id_to_number[source_id]
            
            if format_type == "html":
                formatted_citation = f'<p>[{number}] <a href="{citation.url}" target="_blank" rel="noopener noreferrer">{citation.source}</a></p>'
            else:  # markdown
                formatted_citation = f"[{number}] [{citation.source}]({citation.url})"
                
            formatted_citations.append(formatted_citation)
            citation_objects.append(citation.to_dict())
        
        # If no citations were found in text but we have sources, add a citation for the entire text
        if not seen_source_ids and self.sources:
            first_source_id = list(self.sources.keys())[0]
            citation = self.sources[first_source_id]
            
            # Add a citation marker to the end of the text
            if not clean_text.endswith(" [1]"):
                clean_text = clean_text + " [1]"
                
            if format_type == "html":
                formatted_citation = f'<p>[1] <a href="{citation.url}" target="_blank" rel="noopener noreferrer">{citation.source}</a></p>'
            else:  # markdown
                formatted_citation = f"[1] [{citation.source}]({citation.url})"
                
            formatted_citations.append(formatted_citation)
            citation_objects.append(citation.to_dict())
        
        return {
            "text": clean_text,
            "citations": formatted_citations,
            "citation_objects": citation_objects
        }
