# Part 1: Read the file
with open(r'c:\Users\swara\Downloads\pdf-information-extractor-main\app\componenets\retriver.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the old GENERAL_LLM_PROMPT section
start_marker = 'GENERAL_LLM_PROMPT = """\\nYou are an advanced, intelligent, reliable, and helpful AI assistant'
end_marker = '"""\n'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx + len(start_marker))

if start_idx == -1 or end_idx == -1:
    print('ERROR: Could not find markers')
else:
    print(f'Found old section: start={start_idx}, end={end_idx}')
