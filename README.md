# Agentic Application Skill With Claude - Getting Started

## What This Skill Does

This skill teaches Claude best practices for building agentic applications - AI systems that autonomously complete tasks using tools and reasoning. Whenever you're working on agent development, Claude will reference this skill to provide better guidance.

## How to Use This Skill

### 1. Upload the Skill Files

To activate this skill:
1. Save the `SKILL.md` and `TEMPLATES.md` files to your computer
2. Create a folder structure: `/mnt/skills/user/agentic-app/`
3. Place these files in that folder
4. The skill will be automatically available to Claude in future conversations

### 2. Trigger the Skill Naturally

Just ask Claude questions about agent development:

**Examples that will trigger this skill:**
- "Help me design an agent that can research topics"
- "Create a tool schema for my API"
- "How should I implement an agent loop?"
- "Debug this agent behavior - it's stuck in a loop"
- "Show me how to add memory to my agent"
- "Build a multi-agent system for customer support"

Claude will automatically read the skill before responding.

### 3. Reference Templates

The `TEMPLATES.md` file contains ready-to-use code templates:
- System prompts for different agent types
- Tool definition examples
- Agent loop implementations
- Memory management patterns
- Testing frameworks
- And much more

Just ask: "Show me the template for [specific pattern]"

## Growing This Skill Over Time

The real power of skills comes from continuous improvement. Here's how to evolve this skill:

### After Each Project

1. **Document What Worked**
   - Add successful patterns to SKILL.md
   - Create templates from your working code
   - Note which approaches were most effective

2. **Capture Lessons Learned**
   - Record failures and why they happened
   - Document edge cases you encountered
   - Update best practices based on experience

3. **Add Domain-Specific Sections**
   - Include your specific business context
   - Document your tools and APIs
   - Add your preferred agent behaviors
   - Include terminology and standards

### Example Evolution

**Version 1.0** (Current - General)
```markdown
## Tool Design Best Practices
- Use clear descriptions
- Define parameters explicitly
- Handle errors gracefully
```

**Version 2.0** (After Your First Project)
```markdown
## Tool Design Best Practices
- Use clear descriptions
- Define parameters explicitly
- Handle errors gracefully

## Our Specific Tools
### Salesforce Integration Tool
- Always include account_id parameter
- Rate limit: 100 calls/minute
- Retry logic: 3 attempts with exponential backoff
- Common error: "INVALID_SESSION" - refresh token and retry

### Example from Project Alpha
We found that batching Salesforce updates reduced API calls by 70%.
Template: [link to template]
```

### Maintenance Schedule

**Weekly:** Review recent agent work and capture quick wins
**Monthly:** Do a comprehensive review and update best practices
**Quarterly:** Refactor and reorganize as the skill grows

## Practical Workflow

### Starting a New Agent Project

1. Ask Claude: "I'm building an agent for [purpose]. What should I consider?"
2. Claude reads this skill and provides guidance
3. Implement based on recommendations
4. Test and iterate

### Debugging Existing Agents

1. Ask Claude: "My agent is [problem]. Help me debug."
2. Claude references troubleshooting patterns in the skill
3. Work through solutions systematically
4. Document the fix in the skill for future reference

### Code Review

1. Share your agent code with Claude
2. Ask: "Review this agent implementation against best practices"
3. Claude will check against patterns in this skill
4. Incorporate feedback

## Customization Ideas

Consider adding these sections as your needs grow:

### Business Context
- Your company's domain and terminology
- Common user intents and workflows
- Success criteria and KPIs
- Compliance and security requirements

### Technical Infrastructure
- Your LLM provider and models
- API rate limits and costs
- Database schemas
- Authentication patterns
- Deployment architecture

### Team Standards
- Code style and conventions
- Testing requirements
- Documentation standards
- Review processes

### Performance Benchmarks
- Expected response times
- Acceptable accuracy rates
- Cost per task
- Success rate thresholds

### Case Studies
- Successful agent implementations
- Failed experiments and why
- A/B test results
- User feedback themes

## Quick Reference

**Main Skill File:** `/mnt/skills/user/agentic-app/SKILL.md`
- Comprehensive guide to agent development
- Best practices and patterns
- Troubleshooting advice
- ~400 lines of expert knowledge

**Templates File:** `/mnt/skills/user/agentic-app/TEMPLATES.md`
- Ready-to-use code examples
- System prompts
- Agent loops
- Tool definitions
- Testing frameworks
- ~600 lines of production-ready code

## Tips for Maximum Value

1. **Start Simple:** Use the basic patterns first, add complexity as needed
2. **Iterate Based on Data:** Track what works, update the skill accordingly
3. **Be Specific:** The more specific your skill becomes to your use case, the better
4. **Version Control:** Track changes to understand how your practices evolve
5. **Share Learnings:** If multiple people use this skill, consolidate knowledge
6. **Prune Regularly:** Remove outdated advice as you learn better approaches

## Example Queries to Try

Once you've uploaded this skill, try these:

- "Create a research agent that can search the web and synthesize findings"
- "Show me how to implement tool retry logic with exponential backoff"
- "My agent keeps repeating the same action. How do I detect and prevent loops?"
- "Design a multi-agent system where specialized agents collaborate"
- "Create a system prompt for a customer support agent"
- "Help me implement memory so my agent remembers past conversations"
- "Build a tool schema for my REST API"
- "Show me how to evaluate my agent's performance"

## Next Steps

1. **Save these files** to your preferred location
2. **Upload to Claude** by placing them in `/mnt/skills/user/agentic-app/`
3. **Start building** your first agent with guided support
4. **Update the skill** as you learn and grow
5. **Repeat** - the skill gets better every time you use it

## Need Help?

If you're unsure how to structure something or want to add a new pattern:

1. Ask Claude: "Help me add [topic] to my agentic app skill"
2. Work with Claude to create the new content
3. Update your SKILL.md file with the addition
4. Reference it in future conversations

Remember: This skill is a living document. The more you use and refine it, the more valuable it becomes!

---

**Version:** 1.0  
**Created:** [Date]  
**Last Updated:** [Date]  
**Maintained By:** [Your Name/Team]
