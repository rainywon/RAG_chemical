import re
import os
import sys
import random
from pathlib import Path

def extract_questions(input_file, output_file, count=5000, random_select=False):
    """
    Extract questions from the input file and save them to the output file
    
    Args:
        input_file: Path to source questions file
        output_file: Path to save extracted questions
        count: Number of questions to extract
        random_select: Whether to randomly select questions
    """
    try:
        # Read input file with appropriate encoding
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Find the starting position of the questions list
        match = re.search(r'questions\s*=\s*\[', content)
        if not match:
            print(f"Error: Could not find questions list in {input_file}")
            return False
        
        start_pos = match.end()
        content = content[start_pos:]
        
        # Extract all questions
        questions = []
        quote_pattern = re.compile(r'["\']([^"\']+)["\']')
        
        for match in quote_pattern.finditer(content):
            question = match.group(1).strip()
            if question:
                questions.append(question)
        
        print(f"Found {len(questions)} questions in source file")
        
        # Ensure we have enough questions
        if len(questions) < count:
            print(f"Warning: Only found {len(questions)} questions, fewer than requested {count}")
            count = len(questions)
        
        # Select questions
        if random_select:
            selected_questions = random.sample(questions, count)
        else:
            selected_questions = questions[:count]
        
        # Write selected questions to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("questions = [\n")
            for i, q in enumerate(selected_questions):
                # Format each question with proper escaping and commas
                if i < len(selected_questions) - 1:
                    f.write(f'    "{q}",\n')
                else:
                    f.write(f'    "{q}"\n')
            f.write("]\n")
        
        print(f"Successfully extracted {len(selected_questions)} questions to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup paths
    base_dir = Path(__file__).parent
    input_file = base_dir / "build_lora_data" / "random_22000_questions.py"
    output_file = base_dir / "build_lora_data" / "extracted_5000_questions.py"
    
    # Parse command line arguments
    count = 5000
    random_select = False
    
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid count '{sys.argv[1]}', using default 5000")
    
    if len(sys.argv) > 2 and sys.argv[2].lower() in ('random', 'true', 'yes', 'y', '1'):
        random_select = True
        print("Using random selection")
    
    # Verify input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Extract questions
    success = extract_questions(input_file, output_file, count, random_select)
    if not success:
        sys.exit(1)
    
    print(f"Done! Questions saved to {output_file}") 