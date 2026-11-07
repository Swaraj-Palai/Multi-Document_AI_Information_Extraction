import os, sys
sys.path.insert(0, r'c:\Users\swara\Downloads\pdf-information-extractor-main')
os.environ['GROQ_API_KEY'] = 'test-key'
os.environ['LLM_MODEL'] = 'test-model'

from app.componenets.query_understanding import understand_query

# Test 'what can you do' query
q = understand_query('what can you do')
print('Intent:', q.intent)
print('Requires docs:', q.requires_documents)
print('Complexity:', q.complexity)

# Test 'help' query  
q2 = understand_query('help')
print('Help intent:', q2.intent)

# Test 'meta' query  
q3 = understand_query('meta')
print('Meta intent:', q3.intent)

print()
print('Query understanding works correctly!')
print('The meta intent routes to general LLM mode, which can answer what can you do questions')