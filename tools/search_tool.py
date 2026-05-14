def search_web(query: str) -> str:
    """Mock web search — replace with Tavily or SerpAPI in production."""
    return f"""Search results for: "{query}"

**Result 1: Getting Started with {query}**
A comprehensive guide covering the fundamentals, best practices, and common patterns
when working with {query}. Includes step-by-step tutorials and code examples.
URL: https://docs.example.com/{query.replace(" ", "-").lower()}

**Result 2: {query} — Official Documentation**
The official reference documentation for {query}. Covers installation, configuration,
API reference, and advanced usage patterns.
URL: https://official-docs.example.com/{query.replace(" ", "_").lower()}

**Result 3: {query} Best Practices (2024)**
Community-curated list of best practices and common pitfalls to avoid when
implementing {query} in production environments.
URL: https://community.example.com/best-practices/{query.replace(" ", "-").lower()}

[Note: These are simulated search results. Replace search_web() with a real
search API such as Tavily (pip install tavily-python) for live results.]"""
