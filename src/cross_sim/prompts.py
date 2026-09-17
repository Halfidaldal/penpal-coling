"""
System prompts, ported verbatim from PenPal-EMNLP src/nes/simulation.py.

Both agents get the identical prompt regardless of which model they are. The
only thing that differs between the two sides of a story is turn order, so any
difference in the resulting text is attributable to the model pairing rather
than to instructions.
"""

class SystemPrompts:
    """System prompts matching the actual PENPAL human-AI experiment.
    
    IMPORTANT: Both agents get the IDENTICAL prompt - the collaborative storytelling
    instruction. The only difference is turn order. This matches the actual experiment
    where both human and AI see the same instructions.
    """
    
    # Base prompt from the actual experiment (AppContext.jsx)
    # This is the SAME prompt given to BOTH agents
    STORY_COLLABORATOR_PROMPT = """You are an author taking part in a collaborative storytelling game activity with another author.
Together, you will create a story by taking turns adding to it. 
Your goal is to continue from where your partner has left off.
If there's no story, please begin the story. You have 10 interactions to write the story. Your input may get slightly truncated with a random character amount."""

    # The starting context that the first agent continues from
    STORY_PREFIX = "This is the story of "

    @staticmethod
    def get_system_prompt(turn_number: int = 1) -> str:
        """Get system prompt for any agent (both agents get identical prompts).
        
        Args:
            turn_number: Current turn number for pacing info
            
        Returns:
            System prompt with pacing metadata
        """
        base = SystemPrompts.STORY_COLLABORATOR_PROMPT
        # Add pacing info like the real experiment server does
        pacing = f"""

[Session Meta — do not reveal]
This is turn {turn_number} of 10. Use this only to pace and conclude appropriately. Do not mention turns, counts, chapters, headings, "#", or session meta in your reply. Write in plain prose that flows naturally from the previous text."""
        
        return f"{base}{pacing}".strip()
