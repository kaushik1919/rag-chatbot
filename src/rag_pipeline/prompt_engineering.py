"""
Advanced prompt engineering templates and strategies for RAG systems.
"""
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts for different use cases."""
    QUESTION_ANSWERING = "qa"
    SUMMARIZATION = "summarize"
    ANALYSIS = "analyze"
    CONVERSATION = "chat"
    CODE_GENERATION = "code"
    CREATIVE_WRITING = "creative"


class PromptTemplate:
    """Template for generating structured prompts."""
    
    def __init__(self, template: str, required_vars: List[str], optional_vars: Optional[List[str]] = None):
        self.template = template
        self.required_vars = required_vars
        self.optional_vars = optional_vars or []
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        # Check required variables
        missing_vars = [var for var in self.required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Set default values for optional variables
        for var in self.optional_vars:
            if var not in kwargs:
                kwargs[var] = ""
        
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")


class PromptEngineer:
    """Advanced prompt engineering for RAG systems."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.conversation_history = []
    
    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize prompt templates for different use cases."""
        templates = {}
        
        # Question Answering Template
        templates[PromptType.QUESTION_ANSWERING] = PromptTemplate(
            template="""You are a knowledgeable assistant that provides accurate, helpful answers based on the given context. Follow these guidelines:

1. Base your answer primarily on the provided context
2. If the context doesn't contain sufficient information, clearly state this
3. Provide specific details when available
4. Use a helpful and professional tone

Context Information:
{context}

Question: {question}

Instructions: {instructions}

Answer: Please provide a comprehensive answer based on the context. Be specific and cite relevant information when possible.""",
            required_vars=["context", "question"],
            optional_vars=["instructions"]
        )
        
        # Summarization Template
        templates[PromptType.SUMMARIZATION] = PromptTemplate(
            template="""Please provide a {summary_type} summary of the following content. Focus on the key points and main ideas.

Content to Summarize:
{content}

Summary Length: {length}
Focus Areas: {focus_areas}

Summary:""",
            required_vars=["content"],
            optional_vars=["summary_type", "length", "focus_areas"]
        )
        
        # Analysis Template
        templates[PromptType.ANALYSIS] = PromptTemplate(
            template="""Please analyze the following content according to the specified criteria. Provide a structured analysis with clear insights.

Content for Analysis:
{content}

Analysis Type: {analysis_type}
Specific Criteria: {criteria}

Please provide your analysis in the following structure:
1. Key Findings
2. Detailed Analysis
3. Conclusions
4. Recommendations (if applicable)

Analysis:""",
            required_vars=["content", "analysis_type"],
            optional_vars=["criteria"]
        )
        
        # Conversation Template
        templates[PromptType.CONVERSATION] = PromptTemplate(
            template="""You are a helpful AI assistant engaged in a conversation. Use the provided context to inform your responses while maintaining a natural conversation flow.

Context Information:
{context}

Conversation History:
{history}

Current Message: {message}

Response Guidelines:
- Be conversational and engaging
- Reference context when relevant
- Maintain consistency with previous conversation
- Ask clarifying questions if needed

Response:""",
            required_vars=["message"],
            optional_vars=["context", "history"]
        )
        
        # Code Generation Template
        templates[PromptType.CODE_GENERATION] = PromptTemplate(
            template="""Generate {language} code based on the following requirements and context.

Context/Documentation:
{context}

Requirements:
{requirements}

Programming Language: {language}
Style Guidelines: {style_guidelines}

Please provide:
1. Clean, well-commented code
2. Explanation of the approach
3. Usage examples if appropriate

Code:""",
            required_vars=["requirements", "language"],
            optional_vars=["context", "style_guidelines"]
        )
        
        # Creative Writing Template
        templates[PromptType.CREATIVE_WRITING] = PromptTemplate(
            template="""Create a {content_type} based on the following inspiration and guidelines.

Inspiration/Reference Material:
{context}

Writing Prompt: {prompt}
Style: {style}
Length: {length}
Tone: {tone}

Creative Content:""",
            required_vars=["prompt", "content_type"],
            optional_vars=["context", "style", "length", "tone"]
        )
        
        return templates
    
    def create_prompt(
        self, 
        prompt_type: PromptType, 
        **kwargs
    ) -> str:
        """Create a prompt using the specified template."""
        try:
            if prompt_type not in self.templates:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")
            
            template = self.templates[prompt_type]
            return template.format(**kwargs)
            
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            raise
    
    def create_rag_prompt(
        self,
        question: str,
        context: str,
        prompt_type: PromptType = PromptType.QUESTION_ANSWERING,
        **kwargs
    ) -> str:
        """Create a RAG-optimized prompt."""
        
        # Enhance context with metadata if available
        enhanced_context = self._enhance_context(context, kwargs.get('metadata'))
        
        # Add conversation history if available
        if hasattr(self, 'conversation_history') and self.conversation_history:
            kwargs['history'] = self._format_conversation_history()
        
        # Create base prompt
        base_kwargs = {
            'question': question,
            'context': enhanced_context,
            'message': question,  # For conversation type
            **kwargs
        }
        
        prompt = self.create_prompt(prompt_type, **base_kwargs)
        
        # Apply post-processing enhancements
        prompt = self._apply_prompt_enhancements(prompt, **kwargs)
        
        return prompt
    
    def _enhance_context(self, context: str, metadata: Optional[List[Dict]] = None) -> str:
        """Enhance context with metadata and structure."""
        if not metadata:
            return context
        
        enhanced_parts = []
        context_parts = context.split('\n\n')
        
        for i, part in enumerate(context_parts):
            if i < len(metadata):
                meta = metadata[i]
                source_info = f"[Source: {meta.get('filename', 'Unknown')}]"
                enhanced_parts.append(f"{source_info}\n{part}")
            else:
                enhanced_parts.append(part)
        
        return '\n\n'.join(enhanced_parts)
    
    def _apply_prompt_enhancements(self, prompt: str, **kwargs) -> str:
        """Apply additional prompt enhancements."""
        enhancements = []
        
        # Add reasoning instructions
        if kwargs.get('enable_reasoning', True):
            enhancements.append("Think step by step and explain your reasoning.")
        
        # Add uncertainty handling
        if kwargs.get('handle_uncertainty', True):
            enhancements.append("If you're unsure about any part of your answer, please indicate your level of confidence.")
        
        # Add source citation requirement
        if kwargs.get('require_citations', True):
            enhancements.append("When possible, reference which source material supports your answer.")
        
        if enhancements:
            enhancement_text = "\n\nAdditional Instructions:\n- " + "\n- ".join(enhancements)
            # Insert before the final "Answer:" or "Response:" section
            if "Answer:" in prompt:
                prompt = prompt.replace("Answer:", enhancement_text + "\n\nAnswer:")
            elif "Response:" in prompt:
                prompt = prompt.replace("Response:", enhancement_text + "\n\nResponse:")
            else:
                prompt += enhancement_text
        
        return prompt
    
    def add_to_conversation(self, user_message: str, assistant_response: str) -> None:
        """Add exchange to conversation history."""
        self.conversation_history.append({
            'user': user_message,
            'assistant': assistant_response
        })
        
        # Keep only recent history to manage context length
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for inclusion in prompts."""
        if not self.conversation_history:
            return ""
        
        formatted_history = []
        for exchange in self.conversation_history:
            formatted_history.append(f"User: {exchange['user']}")
            formatted_history.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(formatted_history)
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def create_few_shot_prompt(
        self,
        examples: List[Dict[str, str]],
        question: str,
        context: str,
        instruction: str = "Answer the question based on the context provided."
    ) -> str:
        """Create a few-shot learning prompt with examples."""
        
        prompt_parts = [instruction, ""]
        
        # Add examples
        for i, example in enumerate(examples, 1):
            prompt_parts.extend([
                f"Example {i}:",
                f"Context: {example['context']}",
                f"Question: {example['question']}",
                f"Answer: {example['answer']}",
                ""
            ])
        
        # Add current question
        prompt_parts.extend([
            "Now, please answer this question:",
            f"Context: {context}",
            f"Question: {question}",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)
    
    def create_chain_of_thought_prompt(
        self,
        question: str,
        context: str,
        reasoning_steps: Optional[List[str]] = None
    ) -> str:
        """Create a chain-of-thought prompt for step-by-step reasoning."""
        
        base_prompt = f"""Please answer the following question by thinking through it step by step.

Context: {context}

Question: {question}

Please follow this reasoning process:
1. First, identify the key information in the context relevant to the question
2. Break down the question into smaller parts if needed
3. Analyze each part using the available information
4. Synthesize your findings into a complete answer
5. State your confidence level and any limitations

Step-by-step reasoning:"""
        
        if reasoning_steps:
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(reasoning_steps)])
            base_prompt += f"\n\nSuggested reasoning steps:\n{steps_text}\n\nYour reasoning:"
        
        return base_prompt
    
    def get_template_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available templates."""
        info = {}
        for prompt_type, template in self.templates.items():
            info[prompt_type.value] = {
                'required_variables': template.required_vars,
                'optional_variables': template.optional_vars,
                'description': self._get_template_description(prompt_type)
            }
        return info
    
    def _get_template_description(self, prompt_type: PromptType) -> str:
        """Get description for prompt template."""
        descriptions = {
            PromptType.QUESTION_ANSWERING: "Structured Q&A with context-based responses",
            PromptType.SUMMARIZATION: "Content summarization with configurable focus",
            PromptType.ANALYSIS: "Structured analysis with criteria-based evaluation",
            PromptType.CONVERSATION: "Conversational responses with history awareness",
            PromptType.CODE_GENERATION: "Code generation with documentation context",
            PromptType.CREATIVE_WRITING: "Creative content generation with style control"
        }
        return descriptions.get(prompt_type, "Generic prompt template")
