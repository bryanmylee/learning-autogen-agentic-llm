# Agent Reflection Framework

We'll use the nested chat conversation pattern to realize a sophisticated reflection process.

In this example, we'll have a set of reviewer agents nested within a critic agent as the inner monologue to _reflect_ on a blogpost written by a writer agent.

```python
llm_config = {"model": "gpt-3.5-turbo"}
task = '''
       Write a concise but engaging blogpost about
       DeepLearning.AI. Make sure the blogpost is
       within 100 words.
       '''
```

## Creating the agents for this lesson

```python
import autogen

writer = autogen.AssistantAgent(
    name="Writer",
    system_message="You are a writer. You write engaging and concise "
        "blogposts (with title) on given topics. You must polish your "
        "writing based on the feedback you receive and give a refined "
        "version. Only return your final work without additional comments.",
    llm_config=llm_config,
)

reply = writer.generate_reply(messages=[{"content": task, "role": "user"}])
```

```
Title: Unleashing the Power of Deep Learning with DeepLearning.AI

Delving into the realm of artificial intelligence has never been more accessible, thanks to DeepLearning.AI. Offering top-notch courses created by Andrew Ng, a pioneer in the field, this platform equips learners with the knowledge and skills needed to master deep learning. Whether you're taking your first steps into AI or aiming to advance your career, DeepLearning.AI provides a structured and engaging learning experience. From computer vision to natural language processing, the possibilities are endless. Join the ranks of aspiring AI enthusiasts globally and unlock the potential of deep learning today with DeepLearning.AI.
```

## Adding reflection

Reflection is one of the notably effective agentic design patterns. To implement reflection, we can involve another agent to reflect on and improve on the work of the writer agent.

The reflecting critic agent will initiate a conversation with the writer to incrementally improve its output.

```python
critic = autogen.AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of "
                "the writer and provide constructive "
                "feedback to help improve the quality of the content."
                "You should encourage the writer to stick to facts and "
                "not oversell.",
)

res = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)
```

```
Critic (to Writer):


       Write a concise but engaging blogpost about
       DeepLearning.AI. Make sure the blogpost is
       within 100 words.


--------------------------------------------------------------------------------
Writer (to Critic):

Title: Unleashing the Power of Deep Learning with DeepLearning.AI

Delving into the realm of artificial intelligence has never been more accessible, thanks to DeepLearning.AI. Offering top-notch courses created by Andrew Ng, a pioneer in the field, this platform equips learners with the knowledge and skills needed to master deep learning. Whether you're taking your first steps into AI or aiming to advance your career, DeepLearning.AI provides a structured and engaging learning experience. From computer vision to natural language processing, the possibilities are endless. Join the ranks of aspiring AI enthusiasts globally and unlock the potential of deep learning today with DeepLearning.AI.

--------------------------------------------------------------------------------
Critic (to Writer):

Your blog post is concise and informative, effectively highlighting the benefits of DeepLearning.AI. However, to enhance the quality of the content, I would recommend focusing more on factual information rather than using exaggerated language like "unleashing the power" and "mastering deep learning." It's important to maintain a balanced tone and stick to verifiable facts to build credibility with the readers. Additionally, consider incorporating specific examples or testimonials to support your claims about the platform's effectiveness. Overall, you're on the right track, just remember to emphasize accuracy and provide practical details for the readers' benefit. Keep up the good work!

--------------------------------------------------------------------------------
Writer (to Critic):

Title: Exploring the World of Deep Learning with DeepLearning.AI

Embark on a journey into the intricacies of artificial intelligence through DeepLearning.AI. Developed by renowned expert Andrew Ng, this platform offers a range of expertly crafted courses in various AI domains. Whether you're a beginner or an industry professional, DeepLearning.AI provides structured, high-quality learning. From computer vision to natural language processing, the curriculum covers diverse topics. Join a global community of AI enthusiasts and dive into the depths of deep learning with confidence. Let DeepLearning.AI be your guide in understanding and applying cutting-edge AI concepts.

--------------------------------------------------------------------------------
```

# Nested Chats

Although the reflection above works, it's still very surface level. Often, we want to realize more complex reflections in the inner monologue of the critic.

For example, we might want criticism on different aspects of the work: 1. whether it works well with SEO, 2. whether it will attract organic traffic, 3. whether it has legal and ethical problems, and more.

To handle this, we can use nested chats to act as the inner monologue of an agent.

```python
SEO_reviewer = autogen.AssistantAgent(
    name="SEO Reviewer",
    llm_config=llm_config,
    system_message="You are an SEO reviewer, known for "
        "your ability to optimize content for search engines, "
        "ensuring that it ranks well and attracts organic traffic. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

legal_reviewer = autogen.AssistantAgent(
    name="Legal Reviewer",
    llm_config=llm_config,
    system_message="You are a legal reviewer, known for "
        "your ability to ensure that content is legally compliant "
        "and free from any potential legal issues. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

ethics_reviewer = autogen.AssistantAgent(
    name="Ethics Reviewer",
    llm_config=llm_config,
    system_message="You are an ethics reviewer, known for "
        "your ability to ensure that content is ethically sound "
        "and free from any potential ethical issues. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

meta_reviewer = autogen.AssistantAgent(
    name="Meta Reviewer",
    llm_config=llm_config,
    system_message="You are a meta reviewer, you aggragate and review "
    "the work of other reviewers and give a final suggestion on the content.",
)
```

## Orchestrating the nested chats

We need to set up the initial message properly so that the nested reviewers can get the content to be reviewed. One common way to do this is to get the content from the summary of the author's chat session.

To do this, pass a `message` function for configuration instead. `message` is a function that takes the arguments `(recipient, messages, sender, config)`.

> Since we start the conversation with `critic.initiate_chat(recipient=writer)`, `recipient` represents the writer and contains the blogpost to review.

```python
def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content.
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

review_chats = [
    {
      "recipient": SEO_reviewer,
      "message": reflection_message,
      "summary_method": "reflection_with_llm",
      "summary_args": {
        "summary_prompt":
          "Return review into as JSON object only:"
          "{'Reviewer': '', 'Review': ''}. Here Reviewer should be your role",
      },
      "max_turns": 1
    },
    {
      "recipient": legal_reviewer, "message": reflection_message,
      "summary_method": "reflection_with_llm",
      "summary_args": {
        "summary_prompt":
          "Return review into as JSON object only:"
          "{'Reviewer': '', 'Review': ''}",
       },
      "max_turns": 1
    },
    {
      "recipient": ethics_reviewer, "message": reflection_message,
      "summary_method": "reflection_with_llm",
      "summary_args": {
        "summary_prompt":
          "Return review into as JSON object only:"
          "{'Reviewer': '', 'Review': ''}",
      },
      "max_turns": 1
    },
    {
      "recipient": meta_reviewer,
      "message": "Aggregrate feedback from all reviewers and give final suggestions on the writing.",
      "max_turns": 1
    },
]
```

Once we have a _sequential chat_ list, we register them as nested chats with `ConversableAgent::register_nested_chats(self)`.

```python
critic.register_nested_chats(
    review_chats,
    trigger=writer,
)
```

Initiating the chat between the critic and the writer now produces much more inner monologue.

```python
res = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)
```

```
Critic (to Writer):


       Write a concise but engaging blogpost about
       DeepLearning.AI. Make sure the blogpost is
       within 100 words.


--------------------------------------------------------------------------------
Writer (to Critic):

Title: Unveiling the Power of DeepLearning.AI

Embark on a journey into the groundbreaking world of DeepLearning.AI, where artificial intelligence meets innovation. Offering a cutting-edge platform, DeepLearning.AI equips enthusiasts with the tools to delve into the realm of deep learning. From computer vision to natural language processing, the possibilities are limitless. Sharpen your skills, unravel the complexities of neural networks, and bring your ideas to life. Whether you are a seasoned professional or a curious beginner, DeepLearning.AI opens doors to endless opportunities in the realm of AI. Unleash your potential and redefine the future with DeepLearning.AI.

--------------------------------------------------------------------------------

********************************************************************************
Starting a new chat....

********************************************************************************
Critic (to SEO Reviewer):

Review the following content.


 Title: Unveiling the Power of DeepLearning.AI

Embark on a journey into the groundbreaking world of DeepLearning.AI, where artificial intelligence meets innovation. Offering a cutting-edge platform, DeepLearning.AI equips enthusiasts with the tools to delve into the realm of deep learning. From computer vision to natural language processing, the possibilities are limitless. Sharpen your skills, unravel the complexities of neural networks, and bring your ideas to life. Whether you are a seasoned professional or a curious beginner, DeepLearning.AI opens doors to endless opportunities in the realm of AI. Unleash your potential and redefine the future with DeepLearning.AI.

--------------------------------------------------------------------------------
SEO Reviewer (to Critic):

As an SEO reviewer:

1. Optimize the title by including keyword phrases like "Deep Learning AI courses" or "AI training programs" to improve search visibility.

2. Incorporate relevant keywords throughout the content, such as "neural networks," "computer vision," and "natural language processing," to enhance search engine ranking for those terms.

3. Consider adding a meta description that succinctly summarizes the content using related keywords to attract more organic traffic and improve click-through rates.

--------------------------------------------------------------------------------

********************************************************************************
Starting a new chat....

********************************************************************************
Critic (to Legal Reviewer):

Review the following content.


 Title: Unveiling the Power of DeepLearning.AI

Embark on a journey into the groundbreaking world of DeepLearning.AI, where artificial intelligence meets innovation. Offering a cutting-edge platform, DeepLearning.AI equips enthusiasts with the tools to delve into the realm of deep learning. From computer vision to natural language processing, the possibilities are limitless. Sharpen your skills, unravel the complexities of neural networks, and bring your ideas to life. Whether you are a seasoned professional or a curious beginner, DeepLearning.AI opens doors to endless opportunities in the realm of AI. Unleash your potential and redefine the future with DeepLearning.AI.
Context:
{'Reviewer': 'SEO Specialist', 'Review': 'Title and content are engaging and informative. Recommend optimizing title with relevant keywords and incorporating them throughout the content for better search visibility. Consider adding a meta description with related keywords for improved organic traffic.'}

--------------------------------------------------------------------------------
Legal Reviewer (to Critic):

Legal Reviewer

- Ensure that the use of the term "DeepLearning.AI" does not infringe on any existing trademarks or copyrights.
- Check for any claims or statements implying guaranteed outcomes or results from using DeepLearning.AI, as these could be considered misleading or deceptive.
- Review the content for accuracy in describing the capabilities and limitations of DeepLearning.AI to avoid potential false advertising claims.

--------------------------------------------------------------------------------

********************************************************************************
Starting a new chat....

********************************************************************************
Critic (to Ethics Reviewer):

Review the following content.


 Title: Unveiling the Power of DeepLearning.AI

Embark on a journey into the groundbreaking world of DeepLearning.AI, where artificial intelligence meets innovation. Offering a cutting-edge platform, DeepLearning.AI equips enthusiasts with the tools to delve into the realm of deep learning. From computer vision to natural language processing, the possibilities are limitless. Sharpen your skills, unravel the complexities of neural networks, and bring your ideas to life. Whether you are a seasoned professional or a curious beginner, DeepLearning.AI opens doors to endless opportunities in the realm of AI. Unleash your potential and redefine the future with DeepLearning.AI.
Context:
{'Reviewer': 'SEO Specialist', 'Review': 'Title and content are engaging and informative. Recommend optimizing title with relevant keywords and incorporating them throughout the content for better search visibility. Consider adding a meta description with related keywords for improved organic traffic.'}
{'Reviewer': 'Legal Reviewer', 'Review': '- Ensure that the use of the term "DeepLearning.AI" does not infringe on any existing trademarks or copyrights. - Check for any claims or statements implying guaranteed outcomes or results from using DeepLearning.AI, as these could be considered misleading or deceptive. - Review the content for accuracy in describing the capabilities and limitations of DeepLearning.AI to avoid potential false advertising claims.'}

--------------------------------------------------------------------------------
Ethics Reviewer (to Critic):

As an ethics reviewer, my role is to ensure the content is ethically sound.

- Avoid making exaggerated or unsubstantiated claims about the benefits or outcomes of using DeepLearning.AI to prevent misleading the audience.
- Use inclusive language and imagery to promote diversity and create a welcoming environment for individuals from all backgrounds.
- Provide transparent information about the data collection, privacy policies, and potential risks associated with using DeepLearning.AI to allow users to make informed decisions.

--------------------------------------------------------------------------------

********************************************************************************
Starting a new chat....

********************************************************************************
Critic (to Meta Reviewer):

Aggregrate feedback from all reviewers and give final suggestions on the writing.
Context:
{'Reviewer': 'SEO Specialist', 'Review': 'Title and content are engaging and informative. Recommend optimizing title with relevant keywords and incorporating them throughout the content for better search visibility. Consider adding a meta description with related keywords for improved organic traffic.'}
{'Reviewer': 'Legal Reviewer', 'Review': '- Ensure that the use of the term "DeepLearning.AI" does not infringe on any existing trademarks or copyrights. - Check for any claims or statements implying guaranteed outcomes or results from using DeepLearning.AI, as these could be considered misleading or deceptive. - Review the content for accuracy in describing the capabilities and limitations of DeepLearning.AI to avoid potential false advertising claims.'}
{'Reviewer': 'Ethics Reviewer', 'Review': '- Avoid making exaggerated or unsubstantiated claims about the benefits or outcomes of using DeepLearning.AI to prevent misleading the audience. - Use inclusive language and imagery to promote diversity and create a welcoming environment for individuals from all backgrounds. - Provide transparent information about the data collection, privacy policies, and potential risks associated with using DeepLearning.AI to allow users to make informed decisions.'}

--------------------------------------------------------------------------------
Meta Reviewer (to Critic):

Aggregated Feedback:
- The title and content are engaging, informative, and need optimization for better search visibility.
- Ensure no infringement on trademarks or copyrights with the term "DeepLearning.AI".
- Avoid exaggerated or misleading claims about the benefits of DeepLearning.AI.
- Use inclusive language and imagery to promote diversity.
- Provide transparent information on data collection, privacy policies, and risks associated with DeepLearning.AI.

Final Suggestions:
1. Optimize the title with relevant keywords for improved search visibility.
2. Avoid any potential trademark or copyright infringement issues by checking the use of the term "DeepLearning.AI".
3. Ensure all claims about the outcomes of using DeepLearning.AI are accurate and avoid any misleading or deceptive statements.
4. Incorporate inclusive language and imagery to create a welcoming environment for audiences from diverse backgrounds.
5. Provide transparent information regarding data collection, privacy policies, and risks associated with using DeepLearning.AI to empower users to make informed decisions.

By addressing these suggestions, the content can be enhanced to resonate better with the audience and adhere to legal and ethical standards.

--------------------------------------------------------------------------------
Critic (to Writer):

Aggregated Feedback:
- The title and content are engaging, informative, and need optimization for better search visibility.
- Ensure no infringement on trademarks or copyrights with the term "DeepLearning.AI".
- Avoid exaggerated or misleading claims about the benefits of DeepLearning.AI.
- Use inclusive language and imagery to promote diversity.
- Provide transparent information on data collection, privacy policies, and risks associated with DeepLearning.AI.

Final Suggestions:
1. Optimize the title with relevant keywords for improved search visibility.
2. Avoid any potential trademark or copyright infringement issues by checking the use of the term "DeepLearning.AI".
3. Ensure all claims about the outcomes of using DeepLearning.AI are accurate and avoid any misleading or deceptive statements.
4. Incorporate inclusive language and imagery to create a welcoming environment for audiences from diverse backgrounds.
5. Provide transparent information regarding data collection, privacy policies, and risks associated with using DeepLearning.AI to empower users to make informed decisions.

By addressing these suggestions, the content can be enhanced to resonate better with the audience and adhere to legal and ethical standards.

--------------------------------------------------------------------------------
Writer (to Critic):

Title: Exploring the World of DeepLearning.AI: A Comprehensive Guide

Embark on a transformative journey through the innovative realm of DeepLearning.AI. This cutting-edge platform offers tools to explore the depths of deep learning responsibly. From computer vision to natural language processing, the potential for innovation knows no bounds. Whether you're a seasoned pro or a curious beginner, DeepLearning.AI invites you to unlock endless possibilities in AI. Let's together shape a diverse and inclusive future. Prioritize privacy and transparency by acquainting yourself with data policies and risks associated. Discover, learn, and create with DeepLearning.AI, paving the way for a promising future.

--------------------------------------------------------------------------------
```

## Summarizing the review process

```python
print(res.summary)
```

```
Title: Exploring the World of DeepLearning.AI: A Comprehensive Guide

Embark on a transformative journey through the innovative realm of DeepLearning.AI. This cutting-edge platform offers tools to explore the depths of deep learning responsibly. From computer vision to natural language processing, the potential for innovation knows no bounds. Whether you're a seasoned pro or a curious beginner, DeepLearning.AI invites you to unlock endless possibilities in AI. Let's together shape a diverse and inclusive future. Prioritize privacy and transparency by acquainting yourself with data policies and risks associated. Discover, learn, and create with DeepLearning.AI, paving the way for a promising future.
```
