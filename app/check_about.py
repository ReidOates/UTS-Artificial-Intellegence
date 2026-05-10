import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unittest.mock as mock
sys.modules['tensorflow'] = mock.MagicMock()
from app import app

client = app.test_client()
r = client.get('/about')
print('Status:', r.status_code)
html = r.data.decode('utf-8', errors='replace')

checks = ['Tentang Proyek', 'Tech Stack', 'File Model', 'Tahapan', 'check-circle', 'bi-stack']
for c in checks:
    status = 'OK' if c in html else 'MISSING'
    print(f'  [{status}] {c}')

# Check for Jinja error remnants
if 'jinja2' in html.lower() or 'TemplateSyntaxError' in html:
    print('\nJinja ERROR found in output!')
    idx = html.lower().find('jinja')
    print(html[max(0,idx-100):idx+500])
else:
    print('\nNo Jinja errors detected.')
