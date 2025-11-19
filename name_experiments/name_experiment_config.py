"""Configuration for agent name experiments.

This module defines name categories and generates all combinations for testing
how agent names affect negotiation performance.
"""

# Name categories
GENDER_NAMES = ["Claire Becker", "Hunter Becker"]
AFRICAN_AMERICAN_NAMES = ["Tanisha Washington", "DaShawn Washington"]
CHINESE_NAMES = ["Dawei Li", "Taohua Li"]
COMPANY_NAMES = ["OpenAI", "Anthropic"]


def generate_name_combinations():
    """
    Generate all name pair combinations for experiments.
    
    Returns:
        List of tuples: [(name1, name2), ...] for all combinations
    """
    combinations = []
    
    # Within-category combinations
    # Gender: Claire Becker vs Hunter Becker
    combinations.append(("Claire Becker", "Hunter Becker"))
    
    # African American: Tanisha Washington vs DaShawn Washington
    combinations.append(("Tanisha Washington", "DaShawn Washington"))
    
    # Chinese: Dawei Li vs Taohua Li
    combinations.append(("Dawei Li", "Taohua Li"))
    
    # Cross-category combinations
    # Claire Becker with African American names
    combinations.append(("Claire Becker", "Tanisha Washington"))
    combinations.append(("Claire Becker", "DaShawn Washington"))
    
    # Hunter Becker with African American names
    combinations.append(("Hunter Becker", "Tanisha Washington"))
    combinations.append(("Hunter Becker", "DaShawn Washington"))
    
    # Dawei Li with other names
    combinations.append(("Dawei Li", "Claire Becker"))
    combinations.append(("Dawei Li", "Hunter Becker"))
    combinations.append(("Dawei Li", "Tanisha Washington"))
    combinations.append(("Dawei Li", "DaShawn Washington"))
    
    # Taohua Li with other names
    combinations.append(("Taohua Li", "Claire Becker"))
    combinations.append(("Taohua Li", "Hunter Becker"))
    combinations.append(("Taohua Li", "Tanisha Washington"))
    combinations.append(("Taohua Li", "DaShawn Washington"))
    
    # Company names (only within category, no cross-combinations)
    combinations.append(("OpenAI", "Anthropic"))
    
    return combinations


def get_all_name_pairs():
    """
    Get all name pairs as a list of tuples.
    
    Returns:
        List of tuples: All name pair combinations
    """
    return generate_name_combinations()


def get_name_categories():
    """
    Get all name categories for reference.
    
    Returns:
        Dict mapping category names to lists of names
    """
    return {
        "gender": GENDER_NAMES,
        "african_american": AFRICAN_AMERICAN_NAMES,
        "chinese": CHINESE_NAMES,
        "company": COMPANY_NAMES
    }


# Pre-computed combinations for easy access
ALL_NAME_COMBINATIONS = generate_name_combinations()

