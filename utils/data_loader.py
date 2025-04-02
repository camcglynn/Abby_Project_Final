import os
import csv
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_reproductive_health_data():
    """
    Load reproductive health Q&A data from CSV
    
    Returns:
        list: List of dictionaries containing question-answer pairs
    """
    try:
        logger.info("Loading reproductive health data")
        
        # Try to find the data file
        data_paths = [
            'data/Planned Parenthood Data - Sahana.csv',
            'data/AbortionPPDFAQ.csv'
        ]
        
        data_file = None
        for path in data_paths:
            if os.path.exists(path):
                data_file = path
                break
        
        if not data_file:
            logger.warning("Data file not found. Using fallback data.")
            return _get_fallback_data()
        
        # Load data from CSV
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        
        # Convert to list of dictionaries
        qa_pairs = []
        for _, row in df.iterrows():
            qa_pairs.append({
                'Question': row['Question'],
                'Answer': row['Answer'],
                'Link': row.get('Link', ''),
                'Category': row.get('LLM_Category', 'General')
            })
        
        logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    except Exception as e:
        logger.error(f"Error loading reproductive health data: {str(e)}", exc_info=True)
        return _get_fallback_data()

def _get_fallback_data():
    """
    Provide fallback data if the data file is not found
    
    Returns:
        list: List of basic Q&A pairs
    """
    logger.info("Using fallback data")
    
    return [
        {
            'Question': 'What kinds of emergency contraception are there?',
            'Answer': 'There are 2 ways to prevent pregnancy after unprotected sex: You can get certain IUDs within 120 hours (five days) after having unprotected sex (most effective), or take an emergency contraception pill (morning-after pill) within 120 hours. There are two types of morning-after pills: one with ulipristal acetate (ella, requires prescription) and one with levonorgestrel (Plan B, available over-the-counter).',
            'Link': 'https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception/which-kind-emergency-contraception-should-i-use',
            'Category': 'Birth Control'
        },
        {
            'Question': 'How does emergency contraception work?',
            'Answer': 'Emergency contraception works by preventing or delaying ovulation. Sperm can live in your body up to 6 days after sex, waiting for an egg. Morning-after pills temporarily stop ovulation, preventing the sperm from meeting an egg. They do not cause abortion and won\'t work if you\'re already pregnant.',
            'Link': 'https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception/which-kind-emergency-contraception-should-i-use',
            'Category': 'Birth Control'
        },
        {
            'Question': 'How long do I have to get emergency contraception?',
            'Answer': 'You can use emergency contraception up to 5 days (120 hours) after unprotected sex. It\'s important to act quickly. IUDs and ella are effective throughout all 5 days, while Plan B and other levonorgestrel pills work best within the first 3 days (72 hours).',
            'Link': 'https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception/which-kind-emergency-contraception-should-i-use',
            'Category': 'Birth Control'
        }
    ]

def preprocess_question_answer_pairs(qa_pairs):
    """
    Preprocess question-answer pairs for better model performance
    
    Args:
        qa_pairs (list): List of QA pair dictionaries
    
    Returns:
        list: Processed QA pairs
    """
    processed_pairs = []
    
    # Add important additional QA pairs for common questions
    menstrual_cycle_qa = {
        'Question': 'What is the menstrual cycle?',
        'Answer': '''The menstrual cycle is the monthly hormonal cycle a woman's body goes through to prepare for pregnancy. An average cycle takes about 28 days and occurs in phases: [SOURCE:planned_parenthood]

1. Menstrual phase (Days 1-5): The uterus sheds its lining, resulting in menstrual bleeding or a period. This is when your period happens, and the bleeding usually lasts for 3-7 days. During this time, the body sheds the thickened uterine lining and unfertilized egg from the previous cycle. [SOURCE:planned_parenthood]

2. Follicular phase (Days 1-13): The body prepares to release an egg. Estrogen levels rise and the uterine lining begins to thicken. During this phase, follicle-stimulating hormone (FSH) stimulates the ovaries to produce 5-20 follicles, each containing an immature egg. Usually, only one follicle will mature into an egg, while the others are reabsorbed. [SOURCE:planned_parenthood]

3. Ovulation (Day 14, in a 28-day cycle): The ovary releases a mature egg, which travels through the fallopian tube. This is when pregnancy is most likely to occur. A surge in luteinizing hormone (LH) triggers the release of the egg. Some people experience mild pain called "mittelschmerz" during ovulation. [SOURCE:planned_parenthood]

4. Luteal phase (Days 15-28): If the egg isn't fertilized, hormone levels decrease and the body prepares to shed the uterine lining, starting the cycle again. The empty follicle transforms into a corpus luteum, which releases progesterone to maintain the uterine lining in case of pregnancy. If pregnancy doesn't occur, the corpus luteum breaks down, hormone levels drop, and a new cycle begins. [SOURCE:planned_parenthood]

The length of the menstrual cycle varies from person to person. Some people have shorter cycles (21 days) while others have longer ones (35 days). The cycle length may also vary from month to month for the same person. Tracking your cycle can help you understand your body's patterns and identify any irregularities. [SOURCE:planned_parenthood]''',
        'Link': 'https://www.plannedparenthood.org/learn/health-and-wellness/menstruation',
        'Category': 'Health'
    }
    
    # Add the additional QA pair to the beginning of the list for higher priority
    qa_pairs = [menstrual_cycle_qa] + qa_pairs
    
    for pair in qa_pairs:
        # Clean and process the text
        question = pair['Question'].strip()
        answer = pair['Answer'].strip()
        
        # Skip empty or very short Q&As
        if len(question) < 5 or len(answer) < 10:
            continue
        
        processed_pairs.append({
            'Question': question,
            'Answer': answer,
            'Link': pair.get('Link', ''),
            'Category': pair.get('Category', 'General')
        })
    
    return processed_pairs
