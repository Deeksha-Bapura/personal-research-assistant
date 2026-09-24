from agents.scraper import run as scrape
from agents.summarizer import run as summarize
from agents.fact_checker import run as fact_check
from agents.synthesizer import run as synthesize

sources = scrape('multi-agent AI systems 2025', num_sources=2)
summaries = summarize(sources, 'multi-agent AI systems 2025')
facts = fact_check(summaries, 'multi-agent AI systems 2025')
report = synthesize(summaries, facts, 'multi-agent AI systems 2025')
print(report['report'][:500])
print('...')
print('Confidence:', report['confidence'])
print('Stats:', report['stats'])
