# Agentic Application Development Skill

## Purpose
This skill guides Claude in building, debugging, and enhancing agentic applications - systems where AI agents autonomously perform tasks, make decisions, and interact with tools/APIs.

## When to Use This Skill
- Designing agent architectures
- Implementing agent loops and workflows
- Building tool/function calling systems
- Creating multi-agent systems
- Debugging agent behaviors
- Optimizing agent performance
- Implementing memory and state management

---

## Core Agent Architecture Patterns

### 1. ReAct Pattern (Reasoning + Acting)
The agent alternates between reasoning about what to do and taking actions.

**Structure:**
```
Thought: [Agent reasons about the situation]
Action: [Agent calls a tool/function]
Observation: [Result from the action]
... repeat until task complete ...
Final Answer: [Agent provides result to user]
```

**Best for:** General-purpose agents, research tasks, multi-step reasoning

### 2. Plan-and-Execute Pattern
Agent creates a complete plan upfront, then executes steps sequentially.

**Structure:**
1. Understand task
2. Create detailed plan
3. Execute each step
4. Validate results
5. Return outcome

**Best for:** Well-defined tasks, workflows with dependencies, batch processing

### 3. Reflection Pattern
Agent executes, then critiques its own work and refines.

**Structure:**
1. Initial attempt
2. Self-critique
3. Refinement
4. Validation
5. Repeat if needed

**Best for:** Creative tasks, quality-critical outputs, iterative improvement

### 4. Multi-Agent Collaboration
Multiple specialized agents work together.

**Patterns:**
- **Hierarchical:** Manager agent delegates to worker agents
- **Sequential:** Agents pass work down a pipeline
- **Parallel:** Agents work independently then results merge
- **Debate:** Agents critique each other's outputs

**Best for:** Complex domains, specialized expertise, quality through consensus

---

## Tool/Function Design Best Practices

### Tool Definition Principles
1. **Single Responsibility:** Each tool does one thing well
2. **Clear Naming:** Use verb-noun format (e.g., `search_database`, `send_email`)
3. **Comprehensive Descriptions:** Help the agent understand when and how to use it
4. **Explicit Parameters:** Define all parameters with types and constraints
5. **Error Handling:** Return actionable error messages

### Example Tool Schema
```json
{
  "name": "search_documents",
  "description": "Search internal documents by keyword or semantic similarity. Use this when the user needs information from company documents.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query or question"
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return (1-20)",
        "default": 5
      },
      "filter_by_date": {
        "type": "string",
        "description": "Optional: Filter by date range (ISO format: YYYY-MM-DD)",
        "default": null
      }
    },
    "required": ["query"]
  }
}
```

### Tool Composition Strategies
- **Atomic tools:** Break complex operations into smaller tools
- **Wrapper tools:** Combine multiple API calls into one agent-facing tool
- **Conditional tools:** Only expose tools relevant to current context
- **Progressive disclosure:** Start with basic tools, add advanced ones as needed

---

## Agent Loop Implementation

### Basic Agent Loop
```python
def agent_loop(user_input, max_iterations=10):
    context = initialize_context(user_input)
    
    for iteration in range(max_iterations):
        # Get agent's next action
        response = call_llm(context)
        
        # Check if agent wants to use a tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)
                context.add_observation(result)
        
        # Check if agent is done
        if response.is_final_answer:
            return response.content
        
        # Prevent infinite loops
        if is_stuck(context):
            return fallback_response()
    
    return "Max iterations reached"
```

### Key Loop Components
1. **Context Management:** Track conversation history and observations
2. **Tool Execution:** Safe, validated execution of tool calls
3. **Loop Detection:** Identify when agent is stuck in repetitive behavior
4. **Timeout Handling:** Graceful degradation if loop takes too long
5. **Error Recovery:** Handle tool failures and continue

---

## Memory and State Management

### Types of Memory

**Short-term Memory (Conversation Context)**
- Current task context
- Recent tool results
- User preferences for this session
- Maximum: ~100K tokens for Claude

**Long-term Memory (Persistent Storage)**
- User profile and preferences
- Historical interactions
- Domain knowledge learned over time
- Success/failure patterns

**Working Memory (Scratchpad)**
- Intermediate calculations
- Partial results
- Agent's reasoning traces
- Plans and sub-goals

### Memory Implementation Patterns

**Semantic Memory:**
```python
# Store important facts/learnings
memory_store.add_fact(
    content="User prefers concise responses",
    category="user_preferences",
    timestamp=now(),
    relevance_score=0.9
)

# Retrieve relevant memories
relevant_memories = memory_store.retrieve(
    query="How should I format responses?",
    top_k=5
)
```

**Episodic Memory:**
```python
# Store past interactions
episode = {
    "task": "Data analysis request",
    "actions_taken": [...],
    "tools_used": ["fetch_data", "create_chart"],
    "outcome": "success",
    "user_feedback": "positive"
}
memory_store.add_episode(episode)
```

---

## Prompt Engineering for Agents

### System Prompt Structure

```markdown
# Agent Identity
You are [name], an AI agent specialized in [domain].

# Capabilities
You have access to the following tools:
- [tool_1]: [description]
- [tool_2]: [description]

# Behavior Guidelines
1. Always [expected behavior]
2. Never [prohibited behavior]
3. If uncertain, [fallback behavior]

# Task Approach
When given a task:
1. Understand the goal
2. Break it into steps
3. Execute systematically
4. Validate results

# Output Format
Use this format for your responses:
Thought: [your reasoning]
Action: [tool to use]
Action Input: [parameters]
```

### Prompt Techniques for Better Agent Performance

**Chain of Thought:**
```
"Think step-by-step before acting. Explain your reasoning."
```

**Few-Shot Examples:**
```
Example 1:
User: [request]
Thought: [reasoning]
Action: [tool]
Result: [outcome]

Example 2: ...
```

**Constraints and Guardrails:**
```
"Before using any tool, verify you have all required parameters.
If a tool fails, try an alternative approach.
Maximum 3 attempts per tool before seeking user help."
```

**Self-Critique:**
```
"After completing the task, review your work:
- Did you fully address the user's request?
- Are there any errors or improvements needed?
- Should you take additional actions?"
```

---

## Common Agent Pitfalls and Solutions

### Problem: Agent Gets Stuck in Loops
**Symptoms:** Repeats same action, doesn't progress
**Solutions:**
- Track action history, prevent exact repetitions
- Implement max attempts per tool
- Add loop detection logic
- Provide "give up and ask user" option

### Problem: Agent Hallucinates Tool Calls
**Symptoms:** Calls non-existent tools or uses wrong parameters
**Solutions:**
- Provide clear tool schemas
- Use structured output formats
- Validate tool calls before execution
- Give agent error feedback when calls are invalid

### Problem: Agent Doesn't Know When to Stop
**Symptoms:** Over-optimizes, keeps refining unnecessarily
**Solutions:**
- Define clear completion criteria
- Add "good enough" threshold
- Implement time/cost budgets
- Require explicit "DONE" signal

### Problem: Poor Error Handling
**Symptoms:** Crashes on tool failures, loses context
**Solutions:**
- Return structured error messages
- Teach agent to try alternatives
- Implement exponential backoff for retries
- Provide fallback tools

### Problem: Context Window Overflow
**Symptoms:** Agent loses important information, truncation errors
**Solutions:**
- Summarize old context
- Extract and preserve key facts
- Use external memory/vector stores
- Implement context pruning strategies

---

## Testing and Evaluation

### Test Categories

**Unit Tests (Tool Level)**
- Each tool works correctly in isolation
- Proper error handling
- Parameter validation

**Integration Tests (Agent Level)**
- Agent can complete simple tasks
- Correct tool selection
- Proper error recovery

**End-to-End Tests (System Level)**
- Complex multi-step tasks
- Real-world scenarios
- Edge cases and failures

### Evaluation Metrics

**Task Success Rate:**
- Did agent complete the task correctly?
- Percentage of successful completions

**Efficiency:**
- Number of tool calls needed
- Time to completion
- Token usage

**Quality:**
- Accuracy of final output
- User satisfaction
- Reduction in human intervention needed

**Reliability:**
- Consistency across similar tasks
- Error rate
- Recovery from failures

### Evaluation Approach
```python
def evaluate_agent(test_cases):
    results = []
    for test in test_cases:
        result = {
            "input": test.input,
            "expected": test.expected_output,
            "actual": run_agent(test.input),
            "tools_used": agent.get_tool_log(),
            "iterations": agent.get_iteration_count(),
            "success": None,
            "error": None
        }
        result["success"] = evaluate_output(
            result["actual"], 
            result["expected"]
        )
        results.append(result)
    return analyze_results(results)
```

---

## Agent Observability and Debugging

### Logging Best Practices

**Log Agent Reasoning:**
```python
logger.info("Agent thought", extra={
    "thought": agent_thought,
    "iteration": current_iteration,
    "context_length": len(context)
})
```

**Log Tool Calls:**
```python
logger.info("Tool execution", extra={
    "tool_name": tool_name,
    "parameters": parameters,
    "result": result,
    "execution_time": elapsed_time
})
```

**Log Decision Points:**
```python
logger.info("Agent decision", extra={
    "decision": "continue|stop|fallback",
    "reason": reason,
    "confidence": confidence_score
})
```

### Debugging Techniques

1. **Trace Playback:** Replay exact sequence of events
2. **Thought Visualization:** Display agent's reasoning chain
3. **Tool Call Inspection:** Examine parameters and results
4. **Context Snapshots:** Capture state at each iteration
5. **Counterfactual Analysis:** "What if agent had chosen differently?"

---

## Performance Optimization

### Reduce Latency
- Cache frequent tool results
- Batch API calls when possible
- Use streaming responses
- Implement tool result summaries

### Reduce Costs
- Use smaller models for simple decisions
- Implement early stopping
- Cache and reuse LLM responses
- Summarize long contexts

### Improve Reliability
- Implement retry logic with exponential backoff
- Use multiple fallback strategies
- Validate inputs before tool execution
- Monitor and alert on error rates

---

## Multi-Agent Orchestration

### Coordination Patterns

**Manager-Worker Pattern:**
```python
class ManagerAgent:
    def delegate_task(self, task):
        # Analyze task
        subtasks = self.decompose_task(task)
        
        # Assign to specialized workers
        results = []
        for subtask in subtasks:
            worker = self.select_worker(subtask)
            result = worker.execute(subtask)
            results.append(result)
        
        # Synthesize results
        return self.combine_results(results)
```

**Pipeline Pattern:**
```python
# Sequential processing through specialized agents
data = initial_input
data = research_agent.process(data)
data = analysis_agent.process(data)
data = writing_agent.process(data)
return data
```

**Consensus Pattern:**
```python
# Multiple agents vote/debate
proposals = [agent.propose(task) for agent in agents]
final_decision = consensus_mechanism(proposals)
```

### Communication Protocols
- **Shared Memory:** Agents read/write to common store
- **Message Passing:** Agents send structured messages
- **Event Bus:** Agents publish/subscribe to events
- **Direct Invocation:** Agents call each other's functions

---

## Security and Safety Considerations

### Input Validation
- Sanitize user inputs before processing
- Validate tool parameters against schemas
- Implement rate limiting
- Detect and block injection attacks

### Tool Access Control
- Principle of least privilege
- Role-based access for tools
- Audit logs for sensitive operations
- Require user confirmation for dangerous actions

### Output Filtering
- Check responses for sensitive data leaks
- Filter hallucinated or inappropriate content
- Validate against expected output formats
- Implement content moderation

### Sandboxing
- Execute tools in isolated environments
- Limit file system access
- Restrict network calls
- Implement resource quotas (CPU, memory, time)

---

## Example Agent Implementations

### Research Agent Example
```python
class ResearchAgent:
    def research(self, query):
        # 1. Understand query
        intent = self.analyze_query(query)
        
        # 2. Plan research strategy
        plan = self.create_research_plan(intent)
        
        # 3. Execute searches
        sources = []
        for search_query in plan.queries:
            results = self.search_tool(search_query)
            sources.extend(results)
        
        # 4. Synthesize findings
        synthesis = self.synthesize(sources, intent)
        
        # 5. Validate and return
        if self.validate(synthesis):
            return synthesis
        else:
            return self.refine(synthesis)
```

### Customer Support Agent Example
```python
class SupportAgent:
    def handle_ticket(self, ticket):
        # 1. Classify issue
        category = self.classify(ticket.description)
        
        # 2. Check knowledge base
        solutions = self.search_kb(ticket.description)
        
        # 3. If found, provide solution
        if solutions and solutions[0].confidence > 0.8:
            return self.format_solution(solutions[0])
        
        # 4. Otherwise, escalate
        else:
            return self.escalate_to_human(ticket)
```

---

## Continuous Improvement Strategy

### Collect Feedback
- User satisfaction ratings
- Task completion metrics
- Tool usage patterns
- Error logs and failures

### Analyze Performance
- Identify common failure modes
- Find bottlenecks in agent loop
- Discover underutilized tools
- Detect prompt drift over time

### Iterate on Design
- A/B test prompt variations
- Refine tool descriptions
- Adjust loop parameters
- Update system instructions

### Version Control
- Track prompt versions
- Document changes and rationale
- Measure impact of changes
- Roll back if performance degrades

---

## Domain-Specific Guidance

### [Add Your Domain Here]
As you use this skill, add sections specific to your application:

**Your Business Context:**
- Industry-specific terminology
- Common user intents
- Key workflows
- Success criteria

**Your Tools and APIs:**
- Custom tool descriptions
- API quirks and limitations
- Authentication patterns
- Rate limits and quotas

**Your Agent Behaviors:**
- Preferred reasoning patterns
- Brand voice and tone
- Specific do's and don'ts
- Edge case handling

**Lessons Learned:**
- What worked well
- What failed and why
- Optimization discoveries
- User feedback themes

---

## Quick Reference Checklist

When building an agentic application, ensure you have:

- [ ] Clear agent purpose and scope
- [ ] Well-defined tool schemas with descriptions
- [ ] Robust agent loop with error handling
- [ ] Loop detection and max iterations
- [ ] Comprehensive system prompt
- [ ] Memory/state management strategy
- [ ] Logging and observability
- [ ] Test cases and evaluation metrics
- [ ] Input validation and security measures
- [ ] User feedback mechanism
- [ ] Documentation of agent behaviors
- [ ] Plan for continuous improvement

---

## Resources and Further Reading

**Frameworks and Tools:**
- LangChain, LlamaIndex (agent frameworks)
- AutoGPT, BabyAGI (agent examples)
- OpenAI Assistant API, Anthropic Claude (LLM APIs)

**Research Papers:**
- "ReAct: Synergizing Reasoning and Acting in Language Models"
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "Tool Learning with Foundation Models"

**Best Practices:**
- Anthropic's prompt engineering guide
- OpenAI's function calling best practices
- Agent evaluation frameworks

---

## Version History

**v1.0** - Initial skill creation
- Core architecture patterns
- Tool design best practices
- Agent loop implementation
- Memory management strategies

**[Future versions: Add notes as you refine this skill based on real usage]**

