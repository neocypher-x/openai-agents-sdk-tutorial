import os
import asyncio
from typing import Literal
from rich import print
from agents import (
    Agent, Runner, AsyncOpenAI, OpenAIResponsesModel,
    function_tool
)
from pydantic import BaseModel, Field
from ddgs import DDGS

class SearchResult(BaseModel):
    title: str = Field(..., description='Title of the search result')
    link: str = Field(..., description='Link to the search result')
    snippet: str = Field(..., description='Snippet of the search result')

class SearchResults(BaseModel):
    results: list[SearchResult] = Field(..., description='List of search results')

@function_tool
def search_duckduckgo(query: str, max_results: int = 5, search_type: Literal['web', 'news'] = 'web') -> SearchResults:
    """
    DuckDuckGo tool to search for web or news results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.
        search_type (str): Type of search ('web' or 'news').

    Returns:
        SearchResults: A pydantic model containing the list of search results.
    """
    results = []
    with DDGS() as ddgs:
        if search_type == "news":
            hits = ddgs.news(query, max_results=max_results)
        else:
            hits = ddgs.text(query, max_results=max_results)

        for hit in hits:
            results.append(SearchResult(
                title=hit.get('title', ''),
                link=hit.get('href', ''),
                snippet=hit.get('body', '')
            ))

    return SearchResults(results=results)

def build_agent(output_type):
    llm_model = OpenAIResponsesModel(
        model='gpt-4o-mini',
        openai_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    )
    return Agent(
        name='Web Search Agent',
        model=llm_model,
        output_type=output_type,
        instructions='You are a web search agent. Your job is to search the web for the user and return the results.',
        tools=[search_duckduckgo]
    )

def main():
    # 1) Ask for desired output type
    choice = input("Choose output type ([1] free text, [2] structured SearchResults) [1]: ").strip()
    if choice == '2':
        output_type = SearchResults
    else:
        output_type = str

    # 2) Build the agent with the chosen output type
    agent = build_agent(output_type)

    # 3) Ask for the query
    query = input("Enter your search query: ").strip() or "Search for the latest news about OpenAI Agents SDK."

    # 4) Run synchronously
    response = Runner.run_sync(
        starting_agent=agent,
        input=query,
    )

    # 5) Display results
    print('Token Usage', response.raw_responses[0].usage)
    print(f'{response.last_agent.name}:')

    if output_type is SearchResults:
        # Pretty-print structured results
        for idx, result in enumerate(response.final_output.results, start=1):
            print(f"\nResult {idx}:")
            print(f"Title:   {result.title}")
            print(f"Link:    {result.link}")
            print(f"Snippet: {result.snippet}")
    else:
        # Free-text output
        print(response.final_output)

    for raw_response in response.raw_responses:
        print(raw_response)

if __name__ == "__main__":
    main()
