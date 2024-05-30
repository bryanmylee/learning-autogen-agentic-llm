# Introduction to AutoGen

These are the lesson notes for the DeepLearning.AI course on [AI Agentic Design Patterns with AutoGen](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/1/introduction).

AutoGen is a multi-agent conversation framework to quickly create multiple agents with different roles, personas, and capabilities.

e.g. analysing financial data involves 1. research, 2. writing code to collect data, 3. analyze share prices, 4. synthesize reports. A multi-agent system allows you to defer these tasks to different agents quickly with specializations.

Agents can also iteratively critique, review, and improve themselves until they meet a certain standard.

## Lesson plan

We'll learn about how to train core components by exploring the building agent conversable agent.

We'll create and customize conversations between agents and explore their interactive capabilities.

We'll learn about the multi-agent interaction pattern called Sequential Chats. Each agent will work step-by-step to carry out a list of tasks in sequence.

We'll also explore the agent reflection framework 1. using multiple agents to produce a well-written blogpost, and 2. as tools to create a conversational chess game. This will teach us the nested chat pattern where agents will call other agents and iterate with them before returning results.

We'll learn about a powerful capability tool to use in which we provide a user-defined function (e.g. check if chess moves are legal) and give that to agents to use.

We'll also learn about coding and code execution, where agents will write and run code in a sandboxed environment for example, to compute a result needed for a task.

We'll learn best practices for building multi-agent group chats, illustrated with examples of generating a detailed report. These tasks require planning before execution, which can be handled with a planning agent.
