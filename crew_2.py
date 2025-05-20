from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Use the model name with provider prefix like 'ollama/llama3'
local_llm = Ollama(model="ollama/llama3")


support_agent = Agent(
    role='Customer Support Agent',
    goal='Help users with their product-related queries',
    backstory='You are a helpful assistant trained on the company FAQ and product documents.',
    verbose=True,
    allow_delegation=False,
    llm=local_llm
)

support_task = Task(
    description='Respond to a customer who asks: "How can I track my order placed last week?"',
    agent=support_agent,
    expected_output="A clear and friendly response guiding the customer to track their order using the order ID or customer account."
)

crew = Crew(
    agents=[support_agent],
    tasks=[support_task],
    process=Process.sequential,
    llm=local_llm
)

result = crew.kickoff()
print("\n🔹 CrewAI Response:\n", result)
