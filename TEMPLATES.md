# Agentic Application Templates

## Agent System Prompt Templates

### General Purpose Agent
```markdown
You are an AI agent that helps users complete tasks by reasoning step-by-step and using available tools.

# Available Tools
{tool_descriptions}

# Response Format
For each step, use this format:

Thought: [Explain your reasoning about what to do next]
Action: [Name of the tool to use]
Action Input: [JSON parameters for the tool]

After receiving the observation from a tool, continue with another Thought/Action sequence or provide a Final Answer.

When you have completed the task, respond with:
Final Answer: [Your response to the user]

# Guidelines
- Always think before acting
- Use tools when you need information or to perform actions
- If a tool fails, try an alternative approach
- Ask the user for clarification if the request is ambiguous
- Be concise but complete in your responses
```

### Specialized Research Agent
```markdown
You are a research agent specialized in finding and synthesizing information.

# Your Approach
1. Understand the research question thoroughly
2. Break complex questions into searchable sub-questions
3. Search for information using available tools
4. Synthesize findings from multiple sources
5. Cite your sources
6. Acknowledge limitations or gaps in available information

# Available Tools
{tool_descriptions}

# Quality Standards
- Verify information across multiple sources when possible
- Note conflicting information or uncertainty
- Provide both summary and detailed findings
- Include source URLs or references
```

### Customer Support Agent
```markdown
You are a customer support agent helping users resolve issues.

# Your Goals
- Understand the user's problem clearly
- Provide helpful, accurate solutions
- Maintain a friendly, professional tone
- Escalate to human support when necessary

# Available Tools
{tool_descriptions}

# Response Guidelines
- Ask clarifying questions if the issue is unclear
- Search the knowledge base before escalating
- Provide step-by-step solutions when applicable
- Empathize with user frustrations
- End with asking if the solution worked

# Escalation Criteria
Escalate to human support if:
- The issue requires account access you don't have
- The user explicitly requests human support
- You've tried 3 solutions without success
- The issue involves sensitive information
```

---

## Tool Definition Templates

### API Tool Template
```json
{
  "name": "api_operation_name",
  "description": "Clear description of what this tool does and when to use it. Be specific about the use case.",
  "parameters": {
    "type": "object",
    "properties": {
      "required_param": {
        "type": "string",
        "description": "What this parameter is for and format requirements"
      },
      "optional_param": {
        "type": "integer",
        "description": "Optional parameter with default behavior",
        "default": 10
      },
      "enum_param": {
        "type": "string",
        "enum": ["option1", "option2", "option3"],
        "description": "Parameter with limited valid values"
      }
    },
    "required": ["required_param"]
  },
  "returns": {
    "type": "object",
    "description": "Description of what this tool returns",
    "example": {
      "status": "success",
      "data": {},
      "message": "Operation completed"
    }
  }
}
```

### Database Query Tool
```json
{
  "name": "query_database",
  "description": "Execute a SQL query against the database. Use this to retrieve data based on specific criteria. Do not use for data modification.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "SQL SELECT query to execute. Must be read-only."
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of rows to return",
        "default": 100,
        "maximum": 1000
      }
    },
    "required": ["query"]
  }
}
```

### File Operation Tool
```json
{
  "name": "read_file",
  "description": "Read the contents of a file from the file system. Use this when you need to access file data.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Full path to the file to read"
      },
      "encoding": {
        "type": "string",
        "description": "File encoding (default: utf-8)",
        "default": "utf-8",
        "enum": ["utf-8", "ascii", "latin-1"]
      }
    },
    "required": ["file_path"]
  }
}
```

---

## Agent Loop Implementation Templates

### Basic ReAct Loop (Python)
```python
def react_agent_loop(user_input, tools, max_iterations=10):
    """
    Simple ReAct agent implementation.
    """
    messages = [
        {"role": "system", "content": get_system_prompt(tools)},
        {"role": "user", "content": user_input}
    ]
    
    for iteration in range(max_iterations):
        # Get agent response
        response = call_llm(messages)
        
        # Parse agent's response
        thought, action, action_input = parse_response(response)
        
        # Log the agent's reasoning
        log_agent_step(iteration, thought, action, action_input)
        
        # Check if agent is done
        if action == "Final Answer":
            return action_input
        
        # Execute the tool
        try:
            tool_result = execute_tool(action, action_input, tools)
            observation = f"Observation: {tool_result}"
        except Exception as e:
            observation = f"Error: {str(e)}"
        
        # Add to conversation history
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": observation})
        
        # Check for loops
        if is_repeating(messages):
            return "Agent appears stuck. Please rephrase your request."
    
    return "Maximum iterations reached without completion."


def parse_response(response):
    """Extract thought, action, and input from agent response."""
    thought = extract_between(response, "Thought:", "Action:")
    action = extract_between(response, "Action:", "Action Input:")
    action_input = extract_after(response, "Action Input:")
    return thought.strip(), action.strip(), action_input.strip()


def is_repeating(messages, window=3):
    """Detect if agent is stuck in a loop."""
    if len(messages) < window * 2:
        return False
    
    recent = messages[-window:]
    previous = messages[-window*2:-window]
    
    # Check if recent messages match previous window
    return recent == previous
```

### Async Agent Loop (Python with asyncio)
```python
import asyncio

async def async_agent_loop(user_input, tools, max_iterations=10):
    """
    Async agent loop for concurrent tool execution.
    """
    messages = [
        {"role": "system", "content": get_system_prompt(tools)},
        {"role": "user", "content": user_input}
    ]
    
    for iteration in range(max_iterations):
        response = await call_llm_async(messages)
        
        # Parse potential multiple tool calls
        tool_calls = parse_tool_calls(response)
        
        if not tool_calls:
            # Agent provided final answer
            return extract_final_answer(response)
        
        # Execute tools concurrently
        tasks = [
            execute_tool_async(call.name, call.parameters, tools)
            for call in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format observations
        observations = []
        for call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                obs = f"Tool {call.name} failed: {str(result)}"
            else:
                obs = f"Tool {call.name} returned: {result}"
            observations.append(obs)
        
        # Update conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "\n".join(observations)})
    
    return "Maximum iterations reached."
```

### Plan-and-Execute Agent (Python)
```python
def plan_and_execute_agent(user_input, tools):
    """
    Agent that creates a plan first, then executes it.
    """
    # Phase 1: Planning
    planning_prompt = f"""
    Given this task: {user_input}
    
    Create a step-by-step plan using the available tools: {[t['name'] for t in tools]}
    
    Format your plan as:
    Step 1: [action]
    Step 2: [action]
    ...
    """
    
    plan_response = call_llm(planning_prompt)
    steps = parse_plan(plan_response)
    
    print(f"Plan created with {len(steps)} steps")
    
    # Phase 2: Execution
    results = []
    context = user_input
    
    for i, step in enumerate(steps):
        print(f"Executing step {i+1}/{len(steps)}: {step}")
        
        # Execute step with current context
        execution_prompt = f"""
        Context: {context}
        Current step: {step}
        Previous results: {results}
        
        Execute this step using available tools.
        """
        
        step_result = execute_step(execution_prompt, tools)
        results.append(step_result)
        
        # Update context
        context = f"{context}\nCompleted: {step}\nResult: {step_result}"
    
    # Phase 3: Synthesis
    synthesis_prompt = f"""
    Original task: {user_input}
    Execution results: {results}
    
    Synthesize these results into a final answer.
    """
    
    final_answer = call_llm(synthesis_prompt)
    return final_answer


def parse_plan(plan_text):
    """Extract individual steps from plan."""
    steps = []
    for line in plan_text.split('\n'):
        if line.strip().startswith('Step'):
            step = line.split(':', 1)[1].strip()
            steps.append(step)
    return steps
```

---

## Memory Management Templates

### Simple In-Memory Store
```python
class SimpleMemory:
    """Basic memory for storing conversation history and facts."""
    
    def __init__(self, max_tokens=100000):
        self.messages = []
        self.facts = []
        self.max_tokens = max_tokens
    
    def add_message(self, role, content):
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})
        self._prune_if_needed()
    
    def add_fact(self, fact, category=None):
        """Store an important fact."""
        self.facts.append({
            "content": fact,
            "category": category,
            "timestamp": datetime.now()
        })
    
    def get_context(self):
        """Get current context for the agent."""
        context = {
            "history": self.messages[-10:],  # Last 10 messages
            "relevant_facts": self.facts
        }
        return context
    
    def _prune_if_needed(self):
        """Remove old messages if exceeding token limit."""
        total_tokens = estimate_tokens(self.messages)
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)
            total_tokens = estimate_tokens(self.messages)
```

### Vector Memory Store
```python
class VectorMemory:
    """Memory using embeddings for semantic search."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.memories = []
        self.embeddings = []
    
    def add(self, content, metadata=None):
        """Add memory with embedding."""
        embedding = self.embedding_model.encode(content)
        self.memories.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })
        self.embeddings.append(embedding)
    
    def retrieve(self, query, top_k=5):
        """Retrieve most relevant memories."""
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarity scores
        similarities = cosine_similarity(
            query_embedding, 
            self.embeddings
        )
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return relevant memories
        return [self.memories[i] for i in top_indices]
```

---

## Error Handling Templates

### Retry with Exponential Backoff
```python
import time

def execute_with_retry(func, max_attempts=3, base_delay=1):
    """
    Execute function with exponential backoff retry.
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                # Last attempt failed
                raise
            
            # Calculate delay
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
```

### Graceful Degradation
```python
def robust_tool_execution(tool_name, parameters, tools):
    """
    Execute tool with fallback strategies.
    """
    try:
        # Try primary execution
        result = execute_tool(tool_name, parameters, tools)
        return {"status": "success", "result": result}
    
    except ToolNotFoundException:
        return {
            "status": "error",
            "message": f"Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        }
    
    except InvalidParametersException as e:
        return {
            "status": "error",
            "message": f"Invalid parameters: {str(e)}. Please check the tool schema."
        }
    
    except ToolTimeoutException:
        # Try with reduced scope
        simplified_params = simplify_parameters(parameters)
        try:
            result = execute_tool(tool_name, simplified_params, tools)
            return {
                "status": "partial_success",
                "result": result,
                "message": "Completed with reduced scope due to timeout"
            }
        except:
            return {
                "status": "error",
                "message": "Tool timed out even with simplified parameters"
            }
    
    except Exception as e:
        # Log unexpected error
        log_error(tool_name, parameters, e)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
```

---

## Testing Templates

### Unit Test Template
```python
import pytest

def test_tool_execution():
    """Test that a tool executes correctly."""
    tool = get_tool("search_database")
    
    # Test valid input
    result = tool.execute({"query": "SELECT * FROM users LIMIT 1"})
    assert result["status"] == "success"
    assert len(result["data"]) <= 1
    
    # Test invalid input
    with pytest.raises(InvalidParametersException):
        tool.execute({"query": ""})

def test_agent_single_step():
    """Test agent can complete a simple single-step task."""
    agent = create_agent(tools)
    result = agent.run("What is 2 + 2?")
    assert "4" in result.lower()

def test_error_recovery():
    """Test agent recovers from tool failures."""
    agent = create_agent(tools)
    
    # Simulate tool failure
    with mock.patch('tool.execute', side_effect=Exception("Tool failed")):
        result = agent.run("Use the broken tool")
        
        # Agent should handle error gracefully
        assert result is not None
        assert "error" in result.lower() or "sorry" in result.lower()
```

### Integration Test Template
```python
def test_multi_step_task():
    """Test agent can complete a multi-step task."""
    agent = create_agent(tools)
    
    task = "Find all users who signed up this week and send them a welcome email"
    
    result = agent.run(task)
    
    # Verify agent used correct tools
    tool_log = agent.get_tool_log()
    assert "query_database" in tool_log
    assert "send_email" in tool_log
    
    # Verify outcome
    assert result["status"] == "success"
    assert result["emails_sent"] > 0
```

### Evaluation Template
```python
def evaluate_agent(test_cases):
    """
    Comprehensive agent evaluation.
    """
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "avg_iterations": 0,
        "avg_time": 0,
        "tool_usage": {}
    }
    
    for test in test_cases:
        start_time = time.time()
        
        try:
            # Run agent
            output = agent.run(test["input"])
            
            # Evaluate output
            passed = evaluate_output(
                output, 
                test["expected"],
                test.get("criteria", {})
            )
            
            # Record metrics
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["avg_iterations"] += agent.get_iteration_count()
            results["avg_time"] += time.time() - start_time
            
            # Track tool usage
            for tool in agent.get_tools_used():
                results["tool_usage"][tool] = results["tool_usage"].get(tool, 0) + 1
        
        except Exception as e:
            results["failed"] += 1
            print(f"Test failed with error: {e}")
    
    # Calculate averages
    results["avg_iterations"] /= results["total"]
    results["avg_time"] /= results["total"]
    results["success_rate"] = results["passed"] / results["total"]
    
    return results
```

---

## Monitoring and Logging Templates

### Structured Logging
```python
import logging
import json

class AgentLogger:
    """Structured logging for agent operations."""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"agent.{agent_id}")
    
    def log_thought(self, iteration, thought):
        """Log agent's reasoning."""
        self.logger.info(json.dumps({
            "event": "agent_thought",
            "agent_id": self.agent_id,
            "iteration": iteration,
            "thought": thought,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_tool_call(self, tool_name, parameters, result, duration):
        """Log tool execution."""
        self.logger.info(json.dumps({
            "event": "tool_execution",
            "agent_id": self.agent_id,
            "tool_name": tool_name,
            "parameters": parameters,
            "success": "error" not in str(result).lower(),
            "duration_ms": duration * 1000,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_completion(self, task, outcome, total_iterations, total_time):
        """Log task completion."""
        self.logger.info(json.dumps({
            "event": "task_completion",
            "agent_id": self.agent_id,
            "task": task,
            "outcome": outcome,
            "iterations": total_iterations,
            "duration_s": total_time,
            "timestamp": datetime.now().isoformat()
        }))
```

### Metrics Collection
```python
class AgentMetrics:
    """Collect and expose agent metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_tool_calls": 0,
            "tool_failures": 0,
            "avg_iterations": 0,
            "avg_duration": 0
        }
    
    def record_task(self, success, iterations, duration):
        """Record task completion metrics."""
        self.metrics["total_tasks"] += 1
        
        if success:
            self.metrics["successful_tasks"] += 1
        else:
            self.metrics["failed_tasks"] += 1
        
        # Update rolling averages
        n = self.metrics["total_tasks"]
        self.metrics["avg_iterations"] = (
            (self.metrics["avg_iterations"] * (n-1) + iterations) / n
        )
        self.metrics["avg_duration"] = (
            (self.metrics["avg_duration"] * (n-1) + duration) / n
        )
    
    def record_tool_call(self, success):
        """Record tool call metrics."""
        self.metrics["total_tool_calls"] += 1
        if not success:
            self.metrics["tool_failures"] += 1
    
    def get_summary(self):
        """Get metrics summary."""
        return {
            **self.metrics,
            "success_rate": self.metrics["successful_tasks"] / max(1, self.metrics["total_tasks"]),
            "tool_failure_rate": self.metrics["tool_failures"] / max(1, self.metrics["total_tool_calls"])
        }
```

---

## Configuration Templates

### Agent Configuration (YAML)
```yaml
agent:
  name: "research_agent"
  version: "1.0"
  model: "claude-sonnet-4-20250514"
  
  parameters:
    max_iterations: 10
    temperature: 0.7
    max_tokens: 4000
    timeout_seconds: 300
  
  system_prompt: |
    You are a research agent specialized in finding and synthesizing information.
    Always cite your sources and acknowledge uncertainty.
  
  tools:
    - name: "web_search"
      enabled: true
      config:
        max_results: 10
        timeout: 30
    
    - name: "database_query"
      enabled: true
      config:
        read_only: true
        max_rows: 100
  
  memory:
    type: "vector"
    max_context_tokens: 100000
    retention_days: 30
  
  error_handling:
    max_retries: 3
    retry_delay_seconds: 1
    fallback_strategy: "ask_user"
  
  monitoring:
    log_level: "INFO"
    metrics_enabled: true
    trace_all_calls: false
```

---

## Multi-Agent Templates

### Manager-Worker Pattern
```python
class ManagerAgent:
    """Coordinates work across specialized worker agents."""
    
    def __init__(self, workers):
        self.workers = workers  # Dict of {capability: agent}
    
    def delegate(self, task):
        """Analyze task and delegate to appropriate workers."""
        # Decompose task
        subtasks = self.decompose_task(task)
        
        # Assign to workers
        results = []
        for subtask in subtasks:
            worker = self.select_worker(subtask)
            result = worker.execute(subtask)
            results.append(result)
        
        # Synthesize results
        return self.synthesize(task, results)
    
    def decompose_task(self, task):
        """Break task into subtasks."""
        prompt = f"""
        Task: {task}
        
        Break this into subtasks that can be handled by specialists.
        Available specialists: {list(self.workers.keys())}
        
        Return a list of subtasks with format:
        - [capability]: [subtask description]
        """
        response = self.llm.generate(prompt)
        return self.parse_subtasks(response)
    
    def select_worker(self, subtask):
        """Choose the best worker for a subtask."""
        capability = subtask["capability"]
        return self.workers.get(capability)


class WorkerAgent:
    """Specialized agent for specific capabilities."""
    
    def __init__(self, capability, tools):
        self.capability = capability
        self.tools = tools
    
    def execute(self, subtask):
        """Execute the subtask using specialized tools."""
        return basic_agent_loop(subtask, self.tools)
```

---

## Prompt Optimization Templates

### Few-Shot Examples
```markdown
Here are examples of how to handle similar tasks:

Example 1:
User: Find me articles about AI
Thought: I need to search for recent articles on AI
Action: web_search
Action Input: {"query": "artificial intelligence articles 2024"}
Observation: Found 10 articles about AI developments
Thought: I should summarize the most relevant ones
Final Answer: Here are 3 recent articles about AI: [summaries]

Example 2:
User: What's the weather in Paris?
Thought: I need to get current weather data for Paris
Action: get_weather
Action Input: {"city": "Paris", "country": "France"}
Observation: Temperature: 18°C, Condition: Partly cloudy
Final Answer: The current weather in Paris is 18°C and partly cloudy.

Now handle this new task:
User: {user_request}
```

### Self-Consistency Prompt
```markdown
I will solve this problem using multiple approaches and select the most consistent answer.

Approach 1: [method 1]
[reasoning...]
Answer: [result 1]

Approach 2: [method 2]
[reasoning...]
Answer: [result 2]

Approach 3: [method 3]
[reasoning...]
Answer: [result 3]

Consistency check:
The most common answer across approaches is: [final answer]
Confidence: [high/medium/low] based on agreement
```

