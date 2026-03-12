"""
TCL Compiler - Compiles TCL expressions into executable forms

The TCL compiler transforms parsed expressions into executable bytecode
that can be interpreted by the TCL runtime. It handles:
- Symbol resolution and validation
- Causal relationship compilation
- Constraint compilation
- Mathematical operation compilation
- Cognitive enhancement compilation
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from .tcl_parser import ParsedExpression, TCLParseError
from .tcl_symbols import TCLSymbol, SymbolType, ConceptGraph, CausalityMap

class ByteCodeType(Enum):
    """Types of TCL bytecode operations"""
    LOAD_SYMBOL = "load_symbol"
    STORE_SYMBOL = "store_symbol"
    CAUSAL_LINK = "causal_link"
    CONSTRAINT_APPLY = "constraint_apply"
    MATH_OPERATION = "math_operation"
    CONCEPT_MERGE = "concept_merge"
    COMPRESS = "compress"
    ENHANCE = "enhance"
    PREDICT = "predict"
    JUMP = "jump"
    JUMP_IF = "jump_if"
    RETURN = "return"
    HALT = "halt"

@dataclass
class ByteCodeInstruction:
    """A single TCL bytecode instruction"""
    opcode: ByteCodeType
    operands: List[Any] = field(default_factory=list)
    position: Tuple[int, int] = (0, 0)  # (line, column)
    
    def __str__(self):
        return f"{self.opcode.value} {' '.join(map(str, self.operands))}"

@dataclass
class CompiledTCL:
    """Compiled TCL program with bytecode and metadata"""
    instructions: List[ByteCodeInstruction]
    symbol_table: Dict[str, TCLSymbol]
    metadata: Dict[str, Any]
    entry_point: int = 0
    
    def get_instruction_count(self) -> int:
        return len(self.instructions)
    
    def get_symbol_count(self) -> int:
        return len(self.symbol_table)

class TCLCompilationError(Exception):
    """Exception raised during TCL compilation"""
    pass

class TCLCompiler:
    """Compiler for Thought-Compression Language expressions"""
    
    def __init__(self):
        self.current_symbol_table: Dict[str, TCLSymbol] = {}
        self.constant_pool: Dict[str, Any] = {}
        self.label_map: Dict[str, int] = {}
        self.next_label_id = 0
    
    def compile(self, expressions: List[ParsedExpression], 
                context: Any = None) -> CompiledTCL:
        """
        Compile TCL expressions into bytecode
        
        Args:
            expressions: Parsed TCL expressions
            context: TCL execution context
            
        Returns:
            Compiled TCL program
            
        Raises:
            TCLCompilationError: If compilation fails
        """
        self.current_symbol_table.clear()
        self.constant_pool.clear()
        self.label_map.clear()
        self.next_label_id = 0
        
        instructions = []
        metadata = {
            'expression_count': len(expressions),
            'compilation_timestamp': hashlib.md5(str(expressions).encode()).hexdigest()[:8],
            'complexity_score': self._calculate_complexity(expressions)
        }
        
        # Phase 1: Symbol resolution and table building
        self._build_symbol_table(expressions)
        
        # Phase 2: Generate bytecode
        for expr in expressions:
            expr_instructions = self._compile_expression(expr)
            instructions.extend(expr_instructions)
        
        # Phase 3: Optimize and finalize
        optimized_instructions = self._optimize_instructions(instructions)
        
        return CompiledTCL(
            instructions=optimized_instructions,
            symbol_table=self.current_symbol_table.copy(),
            metadata=metadata
        )
    
    def _calculate_complexity(self, expressions: List[ParsedExpression]) -> float:
        """Calculate the complexity score of expressions"""
        if not expressions:
            return 0.0
        
        total_complexity = 0.0
        
        for expr in expressions:
            expr_complexity = 0.0
            
            # Base complexity by type
            type_weights = {
                'symbol': 1.0,
                'operation': 2.0,
                'causality': 3.0,
                'constraint': 2.5
            }
            
            expr_complexity += type_weights.get(expr.type, 1.0)
            
            # Add complexity based on dependencies
            expr_complexity += len(expr.dependencies) * 0.5
            
            # Add complexity based on content complexity
            if expr.type == 'operation':
                # Operations with multiple operators are more complex
                content_str = str(expr.content)
                expr_complexity += content_str.count('→') * 0.3
                expr_complexity += content_str.count('⟹') * 0.4
            
            total_complexity += expr_complexity
        
        return total_complexity / len(expressions)
    
    def _build_symbol_table(self, expressions: List[ParsedExpression]):
        """Build symbol table from expressions"""
        for expr in expressions:
            if expr.type == 'symbol':
                symbol_name = expr.content.get('symbol', '')
                if symbol_name and symbol_name not in self.current_symbol_table:
                    # Create a new symbol if it doesn't exist
                    symbol = self._create_symbol_from_expression(expr)
                    self.current_symbol_table[symbol_name] = symbol
                    
            elif expr.type == 'causality':
                # Add both cause and effect symbols
                cause = expr.content.get('cause', '')
                effect = expr.content.get('effect', '')
                
                if cause and cause not in self.current_symbol_table:
                    self.current_symbol_table[cause] = self._create_placeholder_symbol(cause)
                
                if effect and effect not in self.current_symbol_table:
                    self.current_symbol_table[effect] = self._create_placeholder_symbol(effect)
    
    def _create_symbol_from_expression(self, expr: ParsedExpression) -> TCLSymbol:
        """Create a TCLSymbol from a parsed expression"""
        symbol_name = expr.content.get('symbol', '')
        symbol_type = self._determine_symbol_type(expr)
        
        return TCLSymbol(
            id=f"generated_{hash(symbol_name)}",
            name=symbol_name,
            type=symbol_type,
            definition=f"Generated from TCL expression: {expr.content}",
            relationships={},
            causal_links=[],
            compression_ratio=0.5,
            cognitive_weight=0.7
        )
    
    def _create_placeholder_symbol(self, name: str) -> TCLSymbol:
        """Create a placeholder symbol for causal relationships"""
        return TCLSymbol(
            id=f"placeholder_{hash(name)}",
            name=name,
            type=SymbolType.CONCEPT,
            definition=f"Placeholder for {name}",
            relationships={},
            causal_links=[],
            compression_ratio=0.3,
            cognitive_weight=0.5
        )
    
    def _determine_symbol_type(self, expr: ParsedExpression) -> SymbolType:
        """Determine the appropriate SymbolType for an expression"""
        symbol_name = expr.content.get('symbol', '').lower()
        
        # Mathematical symbols
        math_symbols = ['∑', '∫', '∂', '∀', '∃', '¬', '∞', '∅']
        if symbol_name in math_symbols:
            return SymbolType.PRIMITIVE
        
        # Cognitive symbols
        cognitive_symbols = ['ψ', 'γ', 'λ', 'ω', 'φ', 'δ']
        if symbol_name in cognitive_symbols:
            return SymbolType.PRIMITIVE
        
        # Causality symbols
        causality_symbols = ['→', '⟹', '⇒']
        if symbol_name in causality_symbols:
            return SymbolType.CAUSALITY
        
        # Default to concept
        return SymbolType.CONCEPT
    
    def _compile_expression(self, expr: ParsedExpression) -> List[ByteCodeInstruction]:
        """Compile a single expression into bytecode"""
        instructions = []
        
        if expr.type == 'symbol':
            instructions = self._compile_symbol(expr)
            
        elif expr.type == 'operation':
            instructions = self._compile_operation(expr)
            
        elif expr.type == 'causality':
            instructions = self._compile_causality(expr)
            
        elif expr.type == 'constraint':
            instructions = self._compile_constraint(expr)
            
        else:
            raise TCLCompilationError(f"Unknown expression type: {expr.type}")
        
        # Add position information
        for instr in instructions:
            instr.position = expr.position
        
        return instructions
    
    def _compile_symbol(self, expr: ParsedExpression) -> List[ByteCodeInstruction]:
        """Compile a symbol expression"""
        symbol_name = expr.content.get('symbol', '')
        
        if not symbol_name:
            raise TCLCompilationError("Empty symbol name")
        
        # Load symbol onto stack
        instructions = [
            ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [symbol_name])
        ]
        
        # If this is a concept, apply compression
        if expr.content.get('type') == 'concept':
            instructions.append(ByteCodeInstruction(ByteCodeType.COMPRESS, []))
        
        return instructions
    
    def _compile_operation(self, expr: ParsedExpression) -> List[ByteCodeInstruction]:
        """Compile an operation expression"""
        left = expr.content.get('left', '')
        operator = expr.content.get('operator', '')
        right = expr.content.get('right', '')
        
        if not all([left, operator, right]):
            raise TCLCompilationError(f"Incomplete operation: {expr.content}")
        
        instructions = []
        
        # Load operands
        instructions.append(ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [left]))
        instructions.append(ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [right]))
        
        # Apply operation based on operator
        if operator in ['→', '⟹', '⇒']:
            # Causal operation
            instructions.append(ByteCodeInstruction(ByteCodeType.CAUSAL_LINK, [operator]))
        elif operator in ['⊥', '∥']:
            # Constraint operation
            instructions.append(ByteCodeInstruction(ByteCodeType.CONSTRAINT_APPLY, [operator]))
        elif operator in ['+', '-', '*', '/', '=', '<', '>']:
            # Mathematical operation
            instructions.append(ByteCodeInstruction(ByteCodeType.MATH_OPERATION, [operator]))
        else:
            # Generic operation - try to merge concepts
            instructions.append(ByteCodeInstruction(ByteCodeType.CONCEPT_MERGE, [operator]))
        
        return instructions
    
    def _compile_causality(self, expr: ParsedExpression) -> List[ByteCodeInstruction]:
        """Compile a causality expression"""
        cause = expr.content.get('cause', '')
        effect = expr.content.get('effect', '')
        operator = expr.content.get('operator', '')
        
        if not all([cause, effect]):
            raise TCLCompilationError(f"Incomplete causality: {expr.content}")
        
        instructions = [
            # Load cause and effect
            ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [cause]),
            ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [effect]),
            
            # Create causal link
            ByteCodeInstruction(ByteCodeType.CAUSAL_LINK, [operator])
        ]
        
        return instructions
    
    def _compile_constraint(self, expr: ParsedExpression) -> List[ByteCodeInstruction]:
        """Compile a constraint expression"""
        instructions = []
        
        if 'constraint' in expr.content:
            # Simple constraint {constraint}
            constraint = expr.content['constraint']
            instructions = [
                ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [constraint]),
                ByteCodeInstruction(ByteCodeType.CONSTRAINT_APPLY, ['{}'])
            ]
            
        else:
            # Binary constraint A ⊥ B or A ∥ B
            left = expr.content.get('left', '')
            right = expr.content.get('right', '')
            operator = expr.content.get('operator', '')
            
            if not all([left, right, operator]):
                raise TCLCompilationError(f"Incomplete constraint: {expr.content}")
            
            instructions = [
                ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [left]),
                ByteCodeInstruction(ByteCodeType.LOAD_SYMBOL, [right]),
                ByteCodeInstruction(ByteCodeType.CONSTRAINT_APPLY, [operator])
            ]
        
        return instructions
    
    def _optimize_instructions(self, instructions: List[ByteCodeInstruction]) -> List[ByteCodeInstruction]:
        """Optimize generated bytecode"""
        optimized = []
        
        i = 0
        while i < len(instructions):
            current = instructions[i]
            
            # Combine consecutive LOAD_SYMBOL operations
            if (current.opcode == ByteCodeType.LOAD_SYMBOL and 
                i + 1 < len(instructions) and 
                instructions[i + 1].opcode == ByteCodeType.LOAD_SYMBOL):
                # This could be optimized further, but for now just add both
                optimized.append(current)
                i += 1
                
            # Remove redundant operations
            elif (current.opcode == ByteCodeType.COMPRESS and
                  i + 1 < len(instructions) and
                  instructions[i + 1].opcode == ByteCodeType.COMPRESS):
                # Skip redundant compression
                i += 1
                
            else:
                optimized.append(current)
                i += 1
        
        # Add return instruction if not present
        if not optimized or optimized[-1].opcode != ByteCodeType.RETURN:
            optimized.append(ByteCodeInstruction(ByteCodeType.RETURN, []))
        
        return optimized
    
    def add_enhancement_instruction(self, enhancement_type: str, parameters: Dict[str, Any] = None) -> ByteCodeInstruction:
        """Add a cognitive enhancement instruction"""
        if parameters is None:
            parameters = {}
        
        return ByteCodeInstruction(
            ByteCodeType.ENHANCE, 
            [enhancement_type, parameters]
        )
    
    def add_prediction_instruction(self, target_symbol: str, depth: int = 3) -> ByteCodeInstruction:
        """Add a prediction instruction for causal analysis"""
        return ByteCodeInstruction(
            ByteCodeType.PREDICT,
            [target_symbol, depth]
        )
    
    def create_label(self, name: str) -> str:
        """Create a unique label for jump instructions"""
        label = f"L{self.next_label_id}_{name}"
        self.next_label_id += 1
        return label
    
    def get_symbol_info(self, symbol_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a compiled symbol"""
        if symbol_name in self.current_symbol_table:
            symbol = self.current_symbol_table[symbol_name]
            return {
                'name': symbol.name,
                'type': symbol.type.value,
                'id': symbol.id,
                'compression_ratio': symbol.compression_ratio,
                'cognitive_weight': symbol.cognitive_weight,
                'relationships': symbol.relationships,
                'causal_links': symbol.causal_links
            }
        return None
    
    def validate_compilation(self, compiled: CompiledTCL) -> bool:
        """Validate a compiled TCL program"""
        try:
            # Check that all symbols referenced exist
            for instr in compiled.instructions:
                if instr.opcode in [ByteCodeType.LOAD_SYMBOL, ByteCodeType.CAUSAL_LINK]:
                    for operand in instr.operands:
                        if isinstance(operand, str) and operand not in compiled.symbol_table:
                            return False
            
            # Check instruction sequence validity
            has_return = any(instr.opcode == ByteCodeType.RETURN for instr in compiled.instructions)
            if not has_return:
                return False
            
            return True
            
        except Exception:
            return False
    
    def generate_assembly_listing(self, compiled: CompiledTCL) -> str:
        """Generate human-readable assembly listing of compiled TCL"""
        lines = []
        
        # Header
        lines.append("TCL Assembly Listing")
        lines.append("=" * 50)
        lines.append(f"Symbols: {compiled.get_symbol_count()}")
        lines.append(f"Instructions: {compiled.get_instruction_count()}")
        lines.append(f"Complexity Score: {compiled.metadata.get('complexity_score', 'N/A')}")
        lines.append()
        
        # Symbol table
        lines.append("Symbol Table:")
        lines.append("-" * 30)
        for name, symbol in compiled.symbol_table.items():
            lines.append(f"  {name}: {symbol.type.value} (weight: {symbol.cognitive_weight:.2f})")
        lines.append()
        
        # Instructions
        lines.append("Instructions:")
        lines.append("-" * 30)
        for i, instr in enumerate(compiled.instructions):
            lines.append(f"  {i:3d}: {str(instr)}")
        
        return "\n".join(lines)

# Example compilation
EXAMPLE_TCL_CODE = [
    "Ψ → Γ",  # Thought causes concept
    "∀x (x → ∞Ψ)",  # Universal causation
    "ΣΨ = Ψ₁ + Ψ₂",  # Superthought composition
]

def compile_example_code():
    """Compile example TCL code and display results"""
    from .tcl_parser import TCLParser
    
    compiler = TCLCompiler()
    parser = TCLParser()
    
    print("TCL Compiler - Example Compilation")
    print("=" * 50)
    
    for i, code in enumerate(EXAMPLE_TCL_CODE, 1):
        print(f"\nExample {i}: {code}")
        print("-" * 40)
        
        try:
            # Parse
            expressions = parser.parse(code)
            print(f"Parsed {len(expressions)} expressions")
            
            # Compile
            compiled = compiler.compile(expressions)
            print(f"Compiled to {compiled.get_instruction_count()} instructions")
            
            # Validate
            is_valid = compiler.validate_compilation(compiled)
            print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
            
            # Show assembly
            print("\nAssembly:")
            print(compiler.generate_assembly_listing(compiled))
            
        except Exception as e:
            print(f"Compilation failed: {e}")

if __name__ == "__main__":
    compile_example_code()