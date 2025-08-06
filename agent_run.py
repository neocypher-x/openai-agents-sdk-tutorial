import os
import asyncio
from rich import print 
from openai.types.responses import ResponseTextDeltaEvent 
from agents import Agent, Runner, AsyncOpenAI, OpenAIResponsesModel
import argparse

def build_user_agent() -> Agent:
    """Helper to configure the LLM model and Agent."""
    llm_model = OpenAIResponsesModel(
        model='gpt-4o-mini',
        openai_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    )
    return Agent(
        name='User Interaction Agent',
        model=llm_model,
        output_type=str,
        instructions='You are a user-facing agent. Your job is to interact with the user and help them with their tasks.',
    )

def run_sync(prompt: str):
    user_agent = build_user_agent()
    response = Runner.run_sync(
        starting_agent=user_agent,
        input=prompt
    )
    print('Token Usage', response.raw_responses[0].usage, '')
    print(f'{response.last_agent.name}:\n{response.final_output}')

async def run_async(prompt: str):
    user_agent = build_user_agent()
    response = await Runner.run(
        starting_agent=user_agent,
        input=prompt
    )
    print('Token Usage', response.raw_responses[0].usage, '')
    print(f'{response.last_agent.name}:\n{response.final_output}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the User Interaction Agent with a prompt")
    parser.add_argument('prompt', nargs='?', help='The prompt to send to the agent')
    args = parser.parse_args()

    # 1) Ask sync vs async first
    mode = input("Run synchronously or asynchronously? (sync/async) [sync]: ").strip().lower()

    # 2) Then get the prompt (either from CLI or via input)
    prompt = args.prompt or input("Enter your prompt: ")

    if mode in ('async', 'a'):
        asyncio.run(run_async(prompt))
    else:
        run_sync(prompt)