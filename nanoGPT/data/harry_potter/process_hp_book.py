import re
import os

# Directory containing all 7 books
books_dir = 'hp_books'

# Output file
output_file = 'harry_potter.txt'

# All 7 book filenames in order
book_files = [
    'Book1.txt',
    'Book2.txt',
    'Book3.txt',
    'Book4.txt',
    'Book5.txt',
    'Book6.txt',
    'Book7.txt',
]

def clean_book(text):
    # Remove page markers like:
    # "Page | 2 Harry Potter and the Philosophers Stone - J.K. Rowling"
    # "Page | 23 Harry Potter and the Chamber of Secrets - J.K. Rowling"
    text = re.sub(r'Page \| \d+.*?J\.K\. Rowling\s*', '', text)
    
    # Remove any leftover standalone page numbers like "| 2" or "Page | 2"
    text = re.sub(r'Page \| \d+', '', text)
    
    # Remove lines that are just the book title (sometimes appears alone)
    text = re.sub(r'^Harry Potter and the .*?$', '', text, flags=re.MULTILINE)
    
    # Remove excessive blank lines (more than 2 in a row) but keep paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

all_books_text = []

for i, filename in enumerate(book_files, start=1):
    filepath = os.path.join(books_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping...")
        continue
    
    print(f"Processing Book {i}: {filename}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean the book
    cleaned = clean_book(text)
    
    print(f"  Done. Characters: {len(cleaned):,}")
    
    all_books_text.append(cleaned)

# Join all books with a clear separator
# <|endoftext|> is GPT-2's special end of document token
# This tells the model "one document ended, another begins"
# so it doesn't mix context across books
separator = '\n\n<|endoftext|>\n\n'
combined = separator.join(all_books_text)

# Save combined file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(combined)

print(f"\nDone! Combined file saved to: {output_file}")
print(f"Total characters: {len(combined):,}")
print(f"Total words (approx): {len(combined.split()):,}")
print(f"Books separated by: <|endoftext|>")