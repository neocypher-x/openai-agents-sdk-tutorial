import os
import asyncio
from rich import print 
from openai.types.responses import ResponseTextDeltaEvent 
from agents import (Agent, Runner, AsyncOpenAI, OpenAIResponsesModel)
import argparse

def run_sync(prompt: str):
    llm_model = OpenAIResponsesModel(model='gpt-4o-mini', openai_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    user_facing_agent = Agent(
        name='User Interaction Agent', 
        model=llm_model, 
        output_type=str,
        instructions='You are a user-facing agent. Your job is to interact with the user and help them their tasks.',
    )

    response = Runner.run_sync(
        starting_agent=user_facing_agent, 
        input=prompt
    )

    print('Token Usage', response.raw_responses[0].usage, '')
    print(f'{response.last_agent.name}:\n{response.final_output}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the User Interaction Agent with a prompt")
    parser.add_argument('prompt', nargs='?', help='The prompt to send to the agent')
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter your prompt: ")

    run_sync(prompt=prompt)