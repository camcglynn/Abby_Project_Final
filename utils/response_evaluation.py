# --- File: utils/response_evaluation.py ---

import re
import logging
# Optional: Install nltk and download vader_lexicon if using sentiment analysis
# `pip install nltk` then run `python -m nltk.downloader vader_lexicon` once
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except (ImportError, LookupError):
    vader_analyzer = None
    VADER_AVAILABLE = False


logger = logging.getLogger(__name__)

# --- Safety Checking ---

# ** Revised Keyword/Pattern Lists **
# REMOVED PLACEHOLDERS LIKE '[racial slur]'. These need real lists managed carefully.
# Added re.escape() around keywords in the check function for safety.

HARMFUL_MEDICAL_ADVICE_PATTERNS = [
    r'\b(induce|cause|start)\s+(miscarriage|abortion)\s+with\s+(herb|vitamin|supplement|method)\b',
    r'\b(bleach|disinfectant|turpentine)\b.*\b(drink|ingest|consume)\b',
    # TODO: Add more specific dangerous medical advice patterns
]

STIGMATIZING_JUDGMENTAL_KEYWORDS = [
    "sinful", "murderer", "irresponsible", "ashamed", "selfish", # Made slightly less specific to catch variations
    "promiscuous", "dirty", "slutty", "unnatural", "punishment", "deserved",
    # TODO: Add/refine keywords based on observed problematic bot outputs
]

MISINFORMATION_PATTERNS = [
    r'\b(abortion|abortions)\s+(causes?|leads? to)\s+(cancer|infertility|mental illness)\b', # Slightly broader verbs
    r'\b(plan b|emergency contraception)\s+(is|are)\s+(an?)?\s*abortifacient(s)?\b',
    r'\bcontracepti(ves?|on)\s+don\'?t\s+work\b',
    # TODO: Add more specific misinformation patterns relevant to your knowledge base
]

COERCIVE_LANGUAGE_PATTERNS = [
    r'\b(you\s+must|you\s+have\s+to)\s+(get an abortion|keep the baby|not tell anyone)\b', # "you should" might be okay sometimes, more context needed
    r'\b(don\'?t\s+tell\s+(your partner|your parents|anyone))\b',
    # TODO: Refine coercive patterns
]

MENTAL_HEALTH_RISK_PATTERNS = [
    # Focus on *promoting* harm
    r'\b(should|better off)\s+(kill yourself|kms|end it all?|not be here)\b',
    r'\b((go\s+)?die|kys)\b',
    # TODO: VERY carefully add patterns, distinguishing from user expression of distress
]

GENERAL_HARMFUL_KEYWORDS = [
    # Use specific lists, avoid placeholders
    "fuck", "shit", "asshole", "bitch", "cunt",
    # TODO: Add actual racial slurs, homophobic slurs etc. CAREFULLY or use external lists/APIs
]

def check_response_safety(text: str) -> bool:
    """
    Checks if the provided text is considered safe for a reproductive
    and mental health context. Uses refined keyword/pattern matching.

    Args:
        text (str): The chatbot response text.

    Returns:
        bool: True if the text is deemed safe, False otherwise.
    """
    if not text:
        logger.debug("Safety check passed: Empty text.")
        return True

    text_lower = text.lower() # Use lowercased only for keyword checks if needed, regex can be case-insensitive

    # Combine all keyword/pattern lists for checking
    all_checks = {
        "HARMFUL_MEDICAL": HARMFUL_MEDICAL_ADVICE_PATTERNS,
        "STIGMA_JUDGMENT": STIGMATIZING_JUDGMENTAL_KEYWORDS,
        "MISINFORMATION": MISINFORMATION_PATTERNS,
        "COERCION": COERCIVE_LANGUAGE_PATTERNS,
        "MENTAL_HEALTH_RISK": MENTAL_HEALTH_RISK_PATTERNS,
        "GENERAL_HARMFUL": GENERAL_HARMFUL_KEYWORDS,
    }

    for category, patterns_or_keywords in all_checks.items():
        for item in patterns_or_keywords:
            try:
                is_keyword = category in ["STIGMA_JUDGMENT", "GENERAL_HARMFUL"] # Categories likely to be single words/simple phrases
                is_pattern = not is_keyword # Assume others are regex patterns

                if is_pattern:
                    # Use regex search for patterns
                    if re.search(item, text, re.IGNORECASE):
                        logger.warning(f"Safety check failed [{category}]: Matched REGEX pattern '{item}' in text: '{text[:100]}...'")
                        return False
                else:
                    # Use more robust keyword matching with word boundaries
                    # re.escape handles special characters within the keyword itself
                    keyword_pattern = r'\b' + re.escape(item) + r'\b'
                    if re.search(keyword_pattern, text_lower): # Check lowercased text for keywords
                        logger.warning(f"Safety check failed [{category}]: Matched KEYWORD '{item}' in text: '{text[:100]}...'")
                        return False

            except re.error as e:
                logger.error(f"Invalid regex pattern in safety checks: '{item}'. Error: {e}")
                # Fail safe on bad regex - better to block potentially good content than allow bad
                return False
            except Exception as e_generic:
                 logger.error(f"Unexpected error during safety check for item '{item}': {e_generic}")
                 return False # Fail safe

    # --- Placeholder for More Advanced Methods ---
    # ... (External APIs, ML models) ...

    logger.debug("Safety check passed.")
    return True


# --- Empathy Calculation ---

EMPATHETIC_PHRASES = [
    "i understand", "that sounds", "it makes sense that you feel", "it's okay to feel",
    "your feelings are valid", "i'm here for you", "i'm here to listen", "how can i help",
    "thank you for sharing", "take your time", "support is available", "reach out",
    "difficult situation", "challenging time", "gentle reminder", "you're not alone",
    "it's understandable", "i can see why", "that must be" # Added more phrases
]
EMPATHETIC_PHRASES_BOOST = 0.12 # Slightly reduced boost per phrase

DISMISSIVE_INVALIDATING_PHRASES = [
    "just relax", "calm down", "don't worry", "it's not a big deal", "you'll get over it",
    "it's simple", "obviously", "you just need to", "stop thinking about it",
    "at least you", "shouldn't feel", "don't be" # Added more phrases
]
DISMISSIVE_PENALTY = 0.20 # Slightly reduced penalty

def calculate_empathy(text: str) -> float:
    """
    Estimates an empathy score for the text (0.0 to 1.0), using refined keyword logic.

    Args:
        text (str): The chatbot response text.

    Returns:
        float: An empathy score. Higher values indicate more perceived empathy.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    score = 0.4 # Start slightly below neutral baseline

    matched_empathetic = []
    matched_dismissive = []

    # Use word boundary regex for more accurate phrase matching
    for phrase in EMPATHETIC_PHRASES:
        try:
            phrase_pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(phrase_pattern, text_lower):
                matched_empathetic.append(phrase)
        except re.error:
             logger.warning(f"Skipping invalid regex derived from empathetic phrase: {phrase}")

    for phrase in DISMISSIVE_INVALIDATING_PHRASES:
        try:
            phrase_pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(phrase_pattern, text_lower):
                matched_dismissive.append(phrase)
        except re.error:
             logger.warning(f"Skipping invalid regex derived from dismissive phrase: {phrase}")


    score += len(matched_empathetic) * EMPATHETIC_PHRASES_BOOST
    score -= len(matched_dismissive) * DISMISSIVE_PENALTY

    sentiment_adjustment = 0.0
    # Optional: Sentiment Analysis Boost/Penalty (VADER)
    if VADER_AVAILABLE and vader_analyzer:
        try:
            vs = vader_analyzer.polarity_scores(text)
            # Use compound score (-1 to +1) for a small adjustment
            sentiment_adjustment = vs['compound'] * 0.10 # Max +/- 0.1 adjustment
            score += sentiment_adjustment
        except Exception as e:
            logger.error(f"Error during VADER analysis for empathy: {e}")

    # Clamp the score to the 0.0 - 1.0 range
    final_score = min(max(score, 0.0), 1.0)

    # Log details for debugging
    log_details = (f"Empathy Score: {final_score:.3f} (Base: 0.4, "
                   f"Pos Phrases: {len(matched_empathetic)} * {EMPATHETIC_PHRASES_BOOST:.2f} = {len(matched_empathetic) * EMPATHETIC_PHRASES_BOOST:.2f}, "
                   f"Neg Phrases: {len(matched_dismissive)} * {DISMISSIVE_PENALTY:.2f} = {-len(matched_dismissive) * DISMISSIVE_PENALTY:.2f}")
    if VADER_AVAILABLE:
         log_details += f", VADER Adj: {sentiment_adjustment:.2f}"
    log_details += ")"
    if matched_empathetic: log_details += f" | Found Pos: {matched_empathetic}"
    if matched_dismissive: log_details += f" | Found Neg: {matched_dismissive}"
    logger.debug(log_details)

    return final_score