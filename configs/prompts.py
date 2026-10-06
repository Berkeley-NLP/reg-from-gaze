"""
Prompt templates for referring expression generation in GazeRL.
"""
from typing import List

# Detailed prompts: descriptive instructions
DETAILED_PROMPTS = [
    "Describe the object in the red box in a way that allows another person to distinguish it from all other objects in the image.",
    "Give a clear and specific description of the object in the red box so that another user can find it without hesitation.",
    "Describe the object in the red box, so that another user can identify the object from other objects.",
    "Describe the object in the red box in a way that allows another person to find that one specific object.",
    "Provide a description of the object in the red box so that another person can recognize and identify that unique item.",
    "Describe the object in the red box so that another person can pinpoint that exact object among others.",
    "Explain the features of the object in the red box that make it stand out, so another person can find it.",
    "Describe the object inside the red box so that someone else can locate that particular item with certainty.",
    "Describe the object in the red box so that another person can find that one specific object.",
    "Give a description of the object in the red box so that another user can identify the exact unique object."
]

# Brief prompts: concise instructions
BRIEF_PROMPTS = [
    "Describe the red-boxed object using the fewest words while ensuring it can be uniquely identified.",
    "Give a minimal description that allows someone to find the exact object in the red box.",
    "Use the least words necessary to ensure the red-boxed object is unmistakably identifiable.",
    "Provide a short yet precise description so the red-boxed object can be uniquely located.",
    "Describe the object in the red box concisely, ensuring it is the only possible match.",
    "Identify the red-boxed object using the fewest words while making it uniquely findable.",
    "Give a brief but unambiguous description that guarantees the red-boxed object can be found.",
    "Provide the shortest possible description that still allows precise identification of the red-boxed object.",
    "Describe the red-boxed object in minimal words while ensuring no confusion with other objects.",
    "Use as few words as possible to describe the red-boxed object in a way that guarantees unique identification."
]

ALL_PROMPTS = DETAILED_PROMPTS + BRIEF_PROMPTS


def get_prompts(prompt_type: str = "all") -> List[str]:
    """
    Retrieve prompt templates by type.
    
    Args:
        prompt_type: One of 'all', 'detailed', or 'brief'.
        
    Returns:
        List of prompt strings.
    """
    if prompt_type == "all":
        return ALL_PROMPTS
    elif prompt_type == "detailed":
        return DETAILED_PROMPTS
    elif prompt_type == "brief":
        return BRIEF_PROMPTS
    else:
        raise ValueError(f"Unknown prompt type '{prompt_type}'. Choose from 'all', 'detailed', or 'brief'.")
