with open(r'c:\Users\swara\Downloads\pdf-information-extractor-main\app\componenets\retriver.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the end of GENERAL_LLM_PROMPT - look for the closing """ after line 437
in_prompt = False
prompt_start = None
prompt_end = None
for i, line in enumerate(lines, 1):
    if i >= 436 and 'GENERAL_LLM_PROMPT' in line:
        in_prompt = True
        prompt_start = i
    if in_prompt and line.strip() == '"""' and i > 500:
        prompt_end = i
        break

if prompt_start and prompt_end:
    print('Prompt starts at line ' + str(prompt_start) + ', ends at line ' + str(prompt_end))
    print('Prompt length: ' + str(prompt_end - prompt_start) + ' lines')
else:
    print('Could not find prompt boundaries')
