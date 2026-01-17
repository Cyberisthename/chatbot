"""
TCL Parser - Converts TCL expressions into structured representations

The TCL parser handles:
- Symbol recognition and validation
- Causal relationship parsing
- Constraint expression parsing
- Mathematical/logical operation parsing

Grammar encodes causality, not syntax - this is fundamental to TCL design
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .tcl_symbols import SymbolType

class TokenType(Enum):
    """Token types for TCL parsing"""
    SYMBOL = "symbol"
    OPERATOR = "operator"
    CAUSALITY = "causality"
    CONSTRAINT = "constraint"
    NUMBER = "number"
    IDENTIFIER = "identifier"
    WHITESPACE = "whitespace"
    END = "end"

@dataclass
class Token:
    """Represents a parsed token"""
    type: TokenType
    value: str
    position: int
    line: int
    column: int

class TCLParseError(Exception):
    """Exception raised during TCL parsing"""
    def __init__(self, message: str, position: int, line: int, column: int):
        super().__init__(f"Parse error at line {line}, column {column}: {message}")
        self.position = position
        self.line = line
        self.column = column

@dataclass
class ParsedExpression:
    """Represents a parsed TCL expression"""
    type: str  # 'symbol', 'causality', 'constraint', 'operation'
    content: Any
    position: Tuple[int, int]  # (start_line, start_column)
    dependencies: List[str] = None  # Symbol IDs this expression depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class SymbolReference:
    """Represents a reference to a TCL symbol"""
    name: str
    symbol_type: SymbolType
    id: Optional[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class TCLParser:
    """Parser for Thought-Compression Language expressions"""
    
    def __init__(self):
        self.current_line = 1
        self.current_column = 1
        self.input = ""
        self.position = 0
        
        # Define symbol patterns
        self.symbol_patterns = [
            # Mathematical symbols
            (r'[∅∞∑∫∂∀∃¬]', TokenType.SYMBOL),
            (r'[ΨΓΛΩΦΔ]', TokenType.SYMBOL),
            (r'[→⟹≡≠⊃∪]', TokenType.SYMBOL),
            
            # Complex symbols (combinations)
            (r'[ΣΨΓΛ∞Ψ∀Γ]', TokenType.SYMBOL),
            
            # Operators
            (r'[+\-*/=<>!]', TokenType.OPERATOR),
            
            # Causality operators
            (r'→|⟹|⇒|⤴|⤵', TokenType.CAUSALITY),
            
            # Constraint markers
            (r'[{}⟂∥⊥]', TokenType.CONSTRAINT),
            
            # Numbers
            (r'\d+\.?\d*', TokenType.NUMBER),
            
            # Identifiers (words)
            (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
        ]
        
        # Causality patterns
        self.causality_patterns = [
            r'(\w+)\s*→\s*(\w+)',  # A causes B
            r'(\w+)\s*⟹\s*(\w+)',  # A implies B
            r'(\w+)\s*⇒\s*(\w+)',  # A therefore B
        ]
        
        # Constraint patterns
        self.constraint_patterns = [
            r'\{([^}]+)\}',           # {constraint}
            r'(\w+)\s*⊥\s*(\w+)',     # A perpendicular B
            r'(\w+)\s*∥\s*(\w+)',     # A parallel B
        ]
    
    def parse(self, tcl_input: str) -> List[ParsedExpression]:
        """
        Parse TCL input into structured expressions
        
        Args:
            tcl_input: TCL expression string to parse
            
        Returns:
            List of parsed expressions
            
        Raises:
            TCLParseError: If parsing fails
        """
        self.input = tcl_input
        self.current_line = 1
        self.current_column = 1
        self.position = 0
        
        expressions = []
        
        # Parse line by line for better error reporting
        lines = tcl_input.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_expressions = self._parse_line(line, line_num)
            expressions.extend(line_expressions)
        
        # Post-process expressions for causality and constraints
        processed_expressions = self._post_process_expressions(expressions)
        
        return processed_expressions
    
    def _parse_line(self, line: str, line_number: int) -> List[ParsedExpression]:
        """Parse a single line of TCL"""
        expressions = []
        
        # Skip empty lines and comments
        line = line.strip()
        if not line or line.startswith('//'):
            return expressions
        
        # Try to match causality patterns first
        for pattern in self.causality_patterns:
            matches = list(re.finditer(pattern, line))
            for match in matches:
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                
                expr = ParsedExpression(
                    type="causality",
                    content={
                        'cause': cause,
                        'effect': effect,
                        'operator': match.group(0)
                    },
                    position=(line_number, match.start() + 1)
                )
                expressions.append(expr)
                
                # Remove matched portion to avoid reprocessing
                line = line[:match.start()] + line[match.end():]
        
        # Try to match constraint patterns
        for pattern in self.constraint_patterns:
            matches = list(re.finditer(pattern, line))
            for match in matches:
                if len(match.groups()) == 1:
                    # {constraint} format
                    constraint = match.group(1).strip()
                    expr = ParsedExpression(
                        type="constraint",
                        content={
                            'constraint': constraint,
                            'operator': match.group(0)
                        },
                        position=(line_number, match.start() + 1)
                    )
                else:
                    # A ⊥ B or A ∥ B format
                    expr = ParsedExpression(
                        type="constraint",
                        content={
                            'left': match.group(1).strip(),
                            'right': match.group(2).strip(),
                            'operator': match.group(0)
                        },
                        position=(line_number, match.start() + 1)
                    )
                expressions.append(expr)
                
                # Remove matched portion
                line = line[:match.start()] + line[match.end():]
        
        # Parse remaining symbols and operations
        remaining_line = line.strip()
        if remaining_line:
            symbol_expressions = self._parse_symbols_and_operations(remaining_line, line_number)
            expressions.extend(symbol_expressions)
        
        return expressions
    
    def _parse_symbols_and_operations(self, line: str, line_number: int) -> List[ParsedExpression]:
        """Parse symbols and operations from remaining line content"""
        expressions = []
        
        # Tokenize the line
        tokens = self._tokenize(line)
        
        # Parse based on token patterns
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token.type == TokenType.SYMBOL:
                # Check if this is a complex operation
                if i + 2 < len(tokens) and tokens[i + 1].type in [TokenType.OPERATOR, TokenType.CAUSALITY]:
                    # Operation or causality
                    operator = tokens[i + 1]
                    right_operand = tokens[i + 2] if i + 2 < len(tokens) else None
                    
                    if right_operand:
                        expr = ParsedExpression(
                            type="operation",
                            content={
                                'left': token.value,
                                'operator': operator.value,
                                'right': right_operand.value
                            },
                            position=(line_number, token.position)
                        )
                        expressions.append(expr)
                        i += 3
                    else:
                        # Unary operation
                        expr = ParsedExpression(
                            type="symbol",
                            content={'symbol': token.value},
                            position=(line_number, token.position)
                        )
                        expressions.append(expr)
                        i += 1
                else:
                    # Simple symbol
                    expr = ParsedExpression(
                        type="symbol",
                        content={'symbol': token.value},
                        position=(line_number, token.position)
                    )
                    expressions.append(expr)
                    i += 1
            
            elif token.type == TokenType.IDENTIFIER:
                # Treat identifiers as concept symbols
                expr = ParsedExpression(
                    type="symbol",
                    content={'symbol': token.value, 'type': 'concept'},
                    position=(line_number, token.position)
                )
                expressions.append(expr)
                i += 1
            
            elif token.type == TokenType.OPERATOR:
                # Skip standalone operators (they should be part of operations)
                i += 1
            
            else:
                # Skip other token types
                i += 1
        
        return expressions
    
    def _tokenize(self, text: str) -> List[Token]:
        """Tokenize text into TCL tokens"""
        tokens = []
        position = 0
        
        while position < len(text):
            matched = False
            
            for pattern, token_type in self.symbol_patterns:
                match = re.match(pattern, text[position:])
                if match:
                    token = Token(
                        type=token_type,
                        value=match.group(0),
                        position=position,
                        line=self.current_line,
                        column=self.current_column + position
                    )
                    tokens.append(token)
                    position += len(match.group(0))
                    matched = True
                    break
            
            if not matched:
                # Skip whitespace
                if text[position].isspace():
                    position += 1
                else:
                    # Unknown character - treat as identifier continuation
                    start = position
                    while position < len(text) and not text[position].isspace():
                        position += 1
                    
                    token = Token(
                        type=TokenType.IDENTIFIER,
                        value=text[start:position],
                        position=start,
                        line=self.current_line,
                        column=self.current_column + start
                    )
                    tokens.append(token)
        
        # Add end token
        tokens.append(Token(
            type=TokenType.END,
            value="",
            position=len(text),
            line=self.current_line,
            column=self.current_column + len(text)
        ))
        
        return tokens
    
    def _post_process_expressions(self, expressions: List[ParsedExpression]) -> List[ParsedExpression]:
        """Post-process parsed expressions to resolve dependencies and relationships"""
        processed = []
        
        for expr in expressions:
            # Add dependencies based on expression type
            if expr.type == "causality":
                expr.dependencies = [expr.content['cause'], expr.content['effect']]
            elif expr.type == "operation":
                expr.dependencies = [expr.content['left'], expr.content['right']]
            elif expr.type == "symbol":
                expr.dependencies = [expr.content['symbol']]
            
            processed.append(expr)
        
        # Sort expressions by dependency order
        processed.sort(key=lambda x: len(x.dependencies))
        
        return processed
    
    def validate_expression(self, expr: ParsedExpression) -> bool:
        """Validate a parsed expression for correctness"""
        try:
            if expr.type == "symbol":
                symbol_name = expr.content.get('symbol', '')
                return len(symbol_name) > 0 and not any(c.isspace() for c in symbol_name)
            
            elif expr.type == "causality":
                cause = expr.content.get('cause', '')
                effect = expr.content.get('effect', '')
                operator = expr.content.get('operator', '')
                return (len(cause) > 0 and len(effect) > 0 and 
                        operator in ['→', '⟹', '⇒'])
            
            elif expr.type == "constraint":
                if 'constraint' in expr.content:
                    constraint = expr.content['constraint']
                    return len(constraint) > 0
                else:
                    left = expr.content.get('left', '')
                    right = expr.content.get('right', '')
                    operator = expr.content.get('operator', '')
                    return (len(left) > 0 and len(right) > 0 and
                            operator in ['⊥', '∥'])
            
            elif expr.type == "operation":
                left = expr.content.get('left', '')
                operator = expr.content.get('operator', '')
                right = expr.content.get('right', '')
                return len(left) > 0 and len(operator) > 0 and len(right) > 0
            
            return False
        
        except Exception:
            return False
    
    def get_expression_info(self, expr: ParsedExpression) -> Dict[str, Any]:
        """Get detailed information about a parsed expression"""
        return {
            'type': expr.type,
            'content': expr.content,
            'position': expr.position,
            'dependencies': expr.dependencies,
            'valid': self.validate_expression(expr),
            'complexity': len(expr.dependencies)
        }

# Example TCL expressions for testing
EXAMPLE_EXPRESSIONS = [
    "Ψ → Γ",  # Thought causes concept
    "Γ ⊥ Λ",  # Concept perpendicular to logic
    "∀x (x → ∞Ψ)",  # Universal causation to infinite thinking
    "ΣΨ = Ψ₁ + Ψ₂ + Ψ₃",  # Superthought as sum of thoughts
    "ΓΛ ⟹ Δ",  # Conceptual logic implies difference
]

def parse_example_expressions():
    """Parse and display example TCL expressions"""
    parser = TCLParser()
    
    print("TCL Parser - Example Expressions")
    print("=" * 40)
    
    for i, expr_text in enumerate(EXAMPLE_EXPRESSIONS, 1):
        print(f"\nExample {i}: {expr_text}")
        print("-" * 30)
        
        try:
            parsed = parser.parse(expr_text)
            
            for j, expr in enumerate(parsed):
                info = parser.get_expression_info(expr)
                print(f"  Expression {j + 1}:")
                print(f"    Type: {info['type']}")
                print(f"    Content: {info['content']}")
                print(f"    Dependencies: {info['dependencies']}")
                print(f"    Valid: {info['valid']}")
                print(f"    Complexity: {info['complexity']}")
                
        except TCLParseError as e:
            print(f"  Parse Error: {e}")

if __name__ == "__main__":
    parse_example_expressions()