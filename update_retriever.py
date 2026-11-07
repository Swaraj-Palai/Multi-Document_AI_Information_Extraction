# Part 1: Read the file and find the old GENERAL_LLM_PROMPT section
with open(r'c:\Users\swara\Downloads\pdf-information-extractor-main\app\componenets\retriver.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the old GENERAL_LLM_PROMPT section
start_marker = 'GENERAL_LLM_PROMPT = """'
end_marker = '"""\n\n\n\nGENERAL_LLM_PROMPT = """'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx == -1 or end_idx == -1:
    print('ERROR: Could not find markers')
else:
    print(f'Found old section: start={start_idx}, end={end_idx}')
    print(f'Section length: {end_idx - start_idx} chars')
