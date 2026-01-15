"""
TCL Runtime - Executes compiled TCL bytecode and performs cognitive enhancements

The TCL runtime interprets compiled TCL bytecode and:
- Manages symbol execution and compression
- Processes causal relationships
- Applies constraints and operations
- Tracks cognitive enhancement metrics
- Provides real-time thinking enhancement

This runtime enables superhuman cognitive capabilities through:
- Symbol compression and concept evolution
- Causal chain prediction and analysis
- Constraint satisfaction and optimization
- Mathematical/logical operation acceleration
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import math

from .tcl_compiler import ByteCodeInstruction, CompiledTCL, ByteCodeType
from .tcl_symbols import TCLSymbol, ConceptGraph, CausalityMap, SymbolType
from .tcl_types import TCLExecutionContext, RuntimeState, ExecutionStack, SymbolStack

class TCLRuntimeError(Exception):
    """Exception raised during TCL runtime execution"""
    pass

class TCLRuntime:
    """Runtime interpreter for compiled TCL bytecode"""
    
    def __init__(self):
        self.execution_stack = ExecutionStack()
        self.symbol_stack = SymbolStack()
        self.variable_table: Dict[str, Any] = {}
        self.runtime_state = RuntimeState.IDLE
        self.instruction_pointer = 0
        self.execution_start_time = 0.0
        self.total_execution_time = 0.0
        self.instruction_count = 0
        self.enhancement_level = 1.0
        self._lock = threading.Lock()
    
    def execute(self, compiled_tcl: CompiledTCL, context: TCLExecutionContext) -> Dict[str, Any]:
        """
        Execute compiled TCL bytecode
        
        Args:
            compiled_tcl: Compiled TCL program
            context: TCL execution context
            
        Returns:
            Execution results and metrics
        """
        with self._lock:
            self.runtime_state = RuntimeState.RUNNING
            self.execution_start_time = time.time()
            self.instruction_pointer = compiled_tcl.entry_point
            self.instruction_count = 0
            self.total_execution_time = 0.0
            
            # Clear stacks and variables
            self.execution_stack.clear()
            self.symbol_stack.clear()
            self.variable_table.clear()
            
            try:
                # Execute instructions
                result = self._execute_instructions(compiled_tcl, context)
                
                # Update enhancement level
                self._update_enhancement_level(context)
                
                return {
                    'result': result,
                    'metrics': {
                        'execution_time': self.total_execution_time,
                        'instruction_count': self.instruction_count,
                        'enhancement_level': self.enhancement_level,
                        'stack_depth': self.execution_stack.size(),
                        'variable_count': len(self.variable_table)
                    },
                    'cognitive_effects': self._analyze_cognitive_effects(context),
                    'symbol_state': self._get_symbol_state(context)
                }
                
            except Exception as e:
                self.runtime_state = RuntimeState.ERROR
                return {
                    'error': str(e),
                    'execution_time': time.time() - self.execution_start_time,
                    'metrics': {
                        'instruction_count': self.instruction_count,
                        'enhancement_level': self.enhancement_level
                    }
                }
            finally:
                self.runtime_state = RuntimeState.IDLE
    
    def _execute_instructions(self, compiled_tcl: CompiledTCL, context: TCLExecutionContext) -> Any:
        """Execute TCL bytecode instructions"""
        instructions = compiled_tcl.instructions
        result = None
        
        while (self.instruction_pointer < len(instructions) and 
               self.runtime_state == RuntimeState.RUNNING):
            
            instruction = instructions[self.instruction_pointer]
            self.instruction_count += 1
            
            # Execute instruction
            instruction_result = self._execute_instruction(instruction, compiled_tcl, context)
            
            if instruction_result is not None:
                result = instruction_result
            
            # Check for halt condition
            if instruction.opcode == ByteCodeType.HALT:
                break
            
            # Move to next instruction
            self.instruction_pointer += 1
            
            # Update execution time
            self.total_execution_time = time.time() - self.execution_start_time
        
        return result
    
    def _execute_instruction(self, instruction: ByteCodeInstruction, 
                           compiled_tcl: CompiledTCL, 
                           context: TCLExecutionContext) -> Any:
        """Execute a single TCL instruction"""
        try:
            if instruction.opcode == ByteCodeType.LOAD_SYMBOL:
                return self._execute_load_symbol(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.STORE_SYMBOL:
                return self._execute_store_symbol(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.CAUSAL_LINK:
                return self._execute_causal_link(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.CONSTRAINT_APPLY:
                return self._execute_constraint_apply(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.MATH_OPERATION:
                return self._execute_math_operation(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.CONCEPT_MERGE:
                return self._execute_concept_merge(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.COMPRESS:
                return self._execute_compress(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.ENHANCE:
                return self._execute_enhance(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.PREDICT:
                return self._execute_predict(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.JUMP:
                return self._execute_jump(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.JUMP_IF:
                return self._execute_jump_if(instruction, compiled_tcl, context)
                
            elif instruction.opcode == ByteCodeType.RETURN:
                return self.execution_stack.pop() if self.execution_stack.size() > 0 else None
                
            elif instruction.opcode == ByteCodeType.HALT:
                self.runtime_state = RuntimeState.HALTED
                return None
                
            else:
                raise TCLRuntimeError(f"Unknown opcode: {instruction.opcode}")
                
        except Exception as e:
            raise TCLRuntimeError(f"Instruction execution failed: {e}")
    
    def _execute_load_symbol(self, instruction: ByteCodeInstruction, 
                           compiled_tcl: CompiledTCL, 
                           context: TCLExecutionContext) -> Any:
        """Execute LOAD_SYMBOL instruction"""
        if not instruction.operands:
            raise TCLRuntimeError("LOAD_SYMBOL requires operand")
        
        symbol_name = instruction.operands[0]
        
        # Try to find symbol in compiled table first
        if symbol_name in compiled_tcl.symbol_table:
            symbol = compiled_tcl.symbol_table[symbol_name]
        else:
            # Try to find in context
            symbol_id = self._find_symbol_in_context(context, symbol_name)
            if symbol_id:
                symbol = context.symbols.symbols[symbol_id]
            else:
                # Create new symbol
                symbol = TCLSymbol(
                    id=f"runtime_{hash(symbol_name)}",
                    name=symbol_name,
                    type=SymbolType.CONCEPT,
                    definition=f"Runtime symbol: {symbol_name}",
                    relationships={},
                    causal_links=[],
                    compression_ratio=0.5,
                    cognitive_weight=0.7
                )
        
        # Push to both stacks
        self.execution_stack.push(symbol)
        self.symbol_stack.push(symbol)
        
        return symbol
    
    def _execute_store_symbol(self, instruction: ByteCodeInstruction, 
                            compiled_tcl: CompiledTCL, 
                            context: TCLExecutionContext) -> Any:
        """Execute STORE_SYMBOL instruction"""
        if not instruction.operands:
            raise TCLRuntimeError("STORE_SYMBOL requires operand")
        
        if self.execution_stack.size() == 0:
            raise TCLRuntimeError("Stack underflow in STORE_SYMBOL")
        
        symbol_name = instruction.operands[0]
        value = self.execution_stack.pop()
        
        # Store in variable table
        self.variable_table[symbol_name] = value
        
        return value
    
    def _execute_causal_link(self, instruction: ByteCodeInstruction, 
                           compiled_tcl: CompiledTCL, 
                           context: TCLExecutionContext) -> Any:
        """Execute CAUSAL_LINK instruction"""
        if self.symbol_stack.size() < 2:
            raise TCLRuntimeError("Insufficient symbols for causal link")
        
        effect = self.symbol_stack.pop()
        cause = self.symbol_stack.pop()
        operator = instruction.operands[0] if instruction.operands else "→"
        
        # Create causal relationship
        strength = self._calculate_causal_strength(cause, effect)
        
        # Add to causality map
        if cause.id not in context.causality.causal_edges:
            context.causality.causal_edges[cause.id] = {}
        
        context.causality.causal_edges[cause.id][effect.id] = strength
        
        # Update cause's causal links
        if effect.id not in cause.causal_links:
            cause.causal_links.append(effect.id)
        
        # Create result symbol
        result = TCLSymbol(
            id=f"causal_result_{int(time.time())}",
            name=f"{cause.name} {operator} {effect.name}",
            type=SymbolType.CAUSALITY,
            definition=f"Causal relationship: {cause.name} {operator} {effect.name}",
            relationships={cause.name: 1.0, effect.name: 1.0},
            causal_links=[cause.id, effect.id],
            compression_ratio=0.8,
            cognitive_weight=(cause.cognitive_weight + effect.cognitive_weight) / 2
        )
        
        self.execution_stack.push(result)
        return result
    
    def _execute_constraint_apply(self, instruction: ByteCodeInstruction, 
                                compiled_tcl: CompiledTCL, 
                                context: TCLExecutionContext) -> Any:
        """Execute CONSTRAINT_APPLY instruction"""
        if self.symbol_stack.size() < 2:
            raise TCLRuntimeError("Insufficient symbols for constraint")
        
        right = self.symbol_stack.pop()
        left = self.symbol_stack.pop()
        operator = instruction.operands[0] if instruction.operands else "{}"
        
        # Apply constraint based on operator
        if operator == "⊥":  # Perpendicular
            constraint_result = self._apply_perpendicular_constraint(left, right)
        elif operator == "∥":  # Parallel
            constraint_result = self._apply_parallel_constraint(left, right)
        elif operator == "{}":  # General constraint
            constraint_result = self._apply_general_constraint(left, right)
        else:
            constraint_result = self._apply_general_constraint(left, right)
        
        self.execution_stack.push(constraint_result)
        return constraint_result
    
    def _execute_math_operation(self, instruction: ByteCodeInstruction, 
                              compiled_tcl: CompiledTCL, 
                              context: TCLExecutionContext) -> Any:
        """Execute MATH_OPERATION instruction"""
        if self.execution_stack.size() < 2:
            raise TCLRuntimeError("Insufficient operands for math operation")
        
        right_val = self.execution_stack.pop()
        left_val = self.execution_stack.pop()
        operator = instruction.operands[0] if instruction.operands else "+"
        
        # Extract numeric values or use cognitive weights
        if hasattr(left_val, 'cognitive_weight'):
            left_num = left_val.cognitive_weight
        else:
            left_num = float(left_val) if isinstance(left_val, (int, float)) else 1.0
        
        if hasattr(right_val, 'cognitive_weight'):
            right_num = right_val.cognitive_weight
        else:
            right_num = float(right_val) if isinstance(right_val, (int, float)) else 1.0
        
        # Perform operation
        if operator == "+":
            result = left_num + right_num
        elif operator == "-":
            result = left_num - right_num
        elif operator == "*":
            result = left_num * right_num
        elif operator == "/":
            result = left_num / right_num if right_num != 0 else 0
        elif operator == "=":
            result = 1.0 if left_num == right_num else 0.0
        elif operator == "<":
            result = 1.0 if left_num < right_num else 0.0
        elif operator == ">":
            result = 1.0 if left_num > right_num else 0.0
        else:
            result = left_num  # Default to left operand
        
        # Create result symbol
        result_symbol = TCLSymbol(
            id=f"math_result_{int(time.time())}",
            name=f"({left_val} {operator} {right_val})",
            type=SymbolType.CONCEPT,
            definition=f"Mathematical operation result",
            relationships={},
            causal_links=[],
            compression_ratio=0.6,
            cognitive_weight=min(1.0, result)
        )
        
        self.execution_stack.push(result_symbol)
        return result_symbol
    
    def _execute_concept_merge(self, instruction: ByteCodeInstruction, 
                             compiled_tcl: CompiledTCL, 
                             context: TCLExecutionContext) -> Any:
        """Execute CONCEPT_MERGE instruction"""
        if self.symbol_stack.size() < 2:
            raise TCLRuntimeError("Insufficient symbols for concept merge")
        
        right = self.symbol_stack.pop()
        left = self.symbol_stack.pop()
        operator = instruction.operands[0] if instruction.operands else "merge"
        
        # Merge concepts based on similarity and relationships
        merged_relationships = left.relationships.copy()
        for key, value in right.relationships.items():
            if key in merged_relationships:
                merged_relationships[key] = (merged_relationships[key] + value) / 2
            else:
                merged_relationships[key] = value
        
        # Create merged symbol
        merged_symbol = TCLSymbol(
            id=f"merged_{int(time.time())}",
            name=f"{left.name}_{operator}_{right.name}",
            type=SymbolType.CONCEPT,
            definition=f"Merged concept: {left.definition} + {right.definition}",
            relationships=merged_relationships,
            causal_links=left.causal_links + right.causal_links,
            compression_ratio=(left.compression_ratio + right.compression_ratio) / 2,
            cognitive_weight=(left.cognitive_weight + right.cognitive_weight) / 2
        )
        
        self.execution_stack.push(merged_symbol)
        return merged_symbol
    
    def _execute_compress(self, instruction: ByteCodeInstruction, 
                        compiled_tcl: CompiledTCL, 
                        context: TCLExecutionContext) -> Any:
        """Execute COMPRESS instruction"""
        if self.symbol_stack.size() == 0:
            raise TCLRuntimeError("No symbol to compress")
        
        symbol = self.symbol_stack.pop()
        
        # Compress the symbol
        compressed = symbol.compress(list(context.symbols.symbols.values()))
        
        # Add to context if not already present
        if compressed.id not in context.symbols.symbols:
            context.symbols.add_symbol(compressed)
        
        self.execution_stack.push(compressed)
        self.symbol_stack.push(compressed)
        
        return compressed
    
    def _execute_enhance(self, instruction: ByteCodeInstruction, 
                       compiled_tcl: CompiledTCL, 
                       context: TCLExecutionContext) -> Any:
        """Execute ENHANCE instruction"""
        if not instruction.operands:
            raise TCLRuntimeError("ENHANCE requires enhancement type")
        
        enhancement_type = instruction.operands[0]
        parameters = instruction.operands[1] if len(instruction.operands) > 1 else {}
        
        # Apply cognitive enhancement
        if enhancement_type == "abstract_reasoning":
            self._enhance_abstract_reasoning(context)
        elif enhancement_type == "pattern_recognition":
            self._enhance_pattern_recognition(context)
        elif enhancement_type == "logical_deduction":
            self._enhance_logical_deduction(context)
        elif enhancement_type == "creative_thinking":
            self._enhance_creative_thinking(context)
        else:
            # General enhancement
            context.metrics.abstract_reasoning_score = min(1.0, context.metrics.abstract_reasoning_score + 0.1)
        
        # Return enhancement result
        enhancement_result = {
            'type': enhancement_type,
            'level': context.metrics.abstract_reasoning_score,
            'timestamp': time.time()
        }
        
        self.execution_stack.push(enhancement_result)
        return enhancement_result
    
    def _execute_predict(self, instruction: ByteCodeInstruction, 
                       compiled_tcl: CompiledTCL, 
                       context: TCLExecutionContext) -> Any:
        """Execute PREDICT instruction"""
        if not instruction.operands:
            raise TCLRuntimeError("PREDICT requires target symbol")
        
        target_symbol = instruction.operands[0]
        depth = instruction.operands[1] if len(instruction.operands) > 1 else 3
        
        # Find target symbol
        target_id = self._find_symbol_in_context(context, target_symbol)
        if not target_id:
            return None
        
        # Generate predictions
        predictions = context.causality.predict_effects(target_id)
        
        # Create prediction result
        prediction_result = {
            'target': target_symbol,
            'predictions': predictions,
            'confidence': sum(strength for _, strength in predictions) / len(predictions) if predictions else 0.0,
            'depth': depth
        }
        
        self.execution_stack.push(prediction_result)
        return prediction_result
    
    def _execute_jump(self, instruction: ByteCodeInstruction, 
                    compiled_tcl: CompiledTCL, 
                    context: TCLExecutionContext) -> Any:
        """Execute JUMP instruction"""
        # For now, JUMP is a no-op in linear execution
        # In a full implementation, this would change instruction_pointer
        return None
    
    def _execute_jump_if(self, instruction: ByteCodeInstruction, 
                       compiled_tcl: CompiledTCL, 
                       context: TCLExecutionContext) -> Any:
        """Execute JUMP_IF instruction"""
        if self.execution_stack.size() == 0:
            raise TCLRuntimeError("No condition for JUMP_IF")
        
        condition = self.execution_stack.pop()
        
        # Simple condition checking
        should_jump = False
        if isinstance(condition, (int, float)):
            should_jump = condition != 0
        elif isinstance(condition, bool):
            should_jump = condition
        elif hasattr(condition, 'cognitive_weight'):
            should_jump = condition.cognitive_weight > 0.5
        
        # For now, just return the condition result
        # In full implementation, would modify instruction_pointer
        return should_jump
    
    def _find_symbol_in_context(self, context: TCLExecutionContext, symbol_name: str) -> Optional[str]:
        """Find symbol ID by name in context"""
        for symbol_id, symbol in context.symbols.symbols.items():
            if symbol.name == symbol_name:
                return symbol_id
        return None
    
    def _calculate_causal_strength(self, cause: TCLSymbol, effect: TCLSymbol) -> float:
        """Calculate the strength of a causal relationship"""
        # Base strength from cognitive weights
        base_strength = (cause.cognitive_weight + effect.cognitive_weight) / 2
        
        # Factor in symbol types
        if cause.type == SymbolType.PRIMITIVE and effect.type == SymbolType.CONCEPT:
            multiplier = 1.2  # Primitives strongly influence concepts
        elif cause.type == SymbolType.CAUSALITY:
            multiplier = 1.1  # Causality symbols have strong influence
        else:
            multiplier = 1.0
        
        return min(1.0, base_strength * multiplier)
    
    def _apply_perpendicular_constraint(self, left: TCLSymbol, right: TCLSymbol) -> TCLSymbol:
        """Apply perpendicular constraint between two symbols"""
        # Perpendicular symbols have minimal relationship
        constraint_symbol = TCLSymbol(
            id=f"perpendicular_{int(time.time())}",
            name=f"{left.name} ⊥ {right.name}",
            type=SymbolType.CONSTRAINT,
            definition=f"Perpendicular constraint: {left.name} ⊥ {right.name}",
            relationships={left.name: 0.1, right.name: 0.1},
            causal_links=[],
            compression_ratio=0.3,
            cognitive_weight=0.5
        )
        return constraint_symbol
    
    def _apply_parallel_constraint(self, left: TCLSymbol, right: TCLSymbol) -> TCLSymbol:
        """Apply parallel constraint between two symbols"""
        # Parallel symbols have strong relationship
        constraint_symbol = TCLSymbol(
            id=f"parallel_{int(time.time())}",
            name=f"{left.name} ∥ {right.name}",
            type=SymbolType.CONSTRAINT,
            definition=f"Parallel constraint: {left.name} ∥ {right.name}",
            relationships={left.name: 0.9, right.name: 0.9},
            causal_links=[],
            compression_ratio=0.8,
            cognitive_weight=0.9
        )
        return constraint_symbol
    
    def _apply_general_constraint(self, left: TCLSymbol, right: TCLSymbol) -> TCLSymbol:
        """Apply general constraint between two symbols"""
        # General constraint with moderate relationship
        constraint_symbol = TCLSymbol(
            id=f"constraint_{int(time.time())}",
            name=f"{{{left.name} {right.name}}}",
            type=SymbolType.CONSTRAINT,
            definition=f"Constraint: {{{left.name} {right.name}}}",
            relationships={left.name: 0.6, right.name: 0.6},
            causal_links=[],
            compression_ratio=0.6,
            cognitive_weight=0.7
        )
        return constraint_symbol
    
    def _enhance_abstract_reasoning(self, context: TCLExecutionContext):
        """Enhance abstract reasoning capabilities"""
        context.metrics.abstract_reasoning_score = min(1.0, context.metrics.abstract_reasoning_score + 0.15)
        context.metrics.cognitive_load = min(1.0, context.metrics.cognitive_load + 0.05)
    
    def _enhance_pattern_recognition(self, context: TCLExecutionContext):
        """Enhance pattern recognition capabilities"""
        # Increase conceptual density for better pattern detection
        context.metrics.conceptual_density = min(1.0, context.metrics.conceptual_density + 0.1)
        context.metrics.thinking_speed = context.metrics.thinking_speed * 1.1
    
    def _enhance_logical_deduction(self, context: TCLExecutionContext):
        """Enhance logical deduction capabilities"""
        context.metrics.causality_depth = min(20, context.metrics.causality_depth + 1)
        context.metrics.abstract_reasoning_score = min(1.0, context.metrics.abstract_reasoning_score + 0.1)
    
    def _enhance_creative_thinking(self, context: TCLExecutionContext):
        """Enhance creative thinking capabilities"""
        context.metrics.cognitive_load = min(1.0, context.metrics.cognitive_load - 0.1)  # Reduce load for creativity
        context.metrics.thinking_speed = context.metrics.thinking_speed * 1.2
    
    def _update_enhancement_level(self, context: TCLExecutionContext):
        """Update the overall enhancement level based on execution metrics"""
        # Base enhancement from abstract reasoning
        base_enhancement = context.metrics.abstract_reasoning_score
        
        # Factor in execution efficiency
        efficiency = min(1.0, self.instruction_count / 100)  # Faster execution = higher efficiency
        efficiency_bonus = efficiency * 0.1
        
        # Factor in symbol compression
        compression_bonus = context.metrics.compression_ratio * 0.1
        
        self.enhancement_level = 1.0 + base_enhancement + efficiency_bonus + compression_bonus
    
    def _analyze_cognitive_effects(self, context: TCLExecutionContext) -> List[str]:
        """Analyze the cognitive effects of the execution"""
        effects = []
        
        if context.metrics.abstract_reasoning_score > 0.7:
            effects.append("Enhanced abstract reasoning capabilities")
        
        if context.metrics.conceptual_density > 0.6:
            effects.append("Increased conceptual connectivity")
        
        if context.metrics.causality_depth > 5:
            effects.append("Deeper causal understanding")
        
        if self.enhancement_level > 1.5:
            effects.append("Significant cognitive enhancement achieved")
        
        if self.execution_stack.size() > 10:
            effects.append("High cognitive complexity processing")
        
        return effects
    
    def _get_symbol_state(self, context: TCLExecutionContext) -> Dict[str, Any]:
        """Get the current state of symbols after execution"""
        active_symbols = []
        
        # Get symbols from stacks
        for symbol in self.symbol_stack.symbols:
            active_symbols.append({
                'name': symbol.name,
                'type': symbol.type.value,
                'weight': symbol.cognitive_weight,
                'compression': symbol.compression_ratio
            })
        
        # Get most recent symbols from context
        recent_symbols = list(context.symbols.symbols.values())[-5:]  # Last 5 symbols
        
        return {
            'active_symbols': active_symbols,
            'total_symbols': len(context.symbols.symbols),
            'recent_symbols': [
                {
                    'name': symbol.name,
                    'type': symbol.type.value,
                    'weight': symbol.cognitive_weight
                } for symbol in recent_symbols
            ]
        }

# Example runtime execution
def demonstrate_tcl_runtime():
    """Demonstrate TCL runtime execution"""
    from .tcl_compiler import TCLCompiler
    from .tcl_parser import TCLParser
    from .tcl_engine import TCLExecutionContext, CognitiveMetrics
    
    # Create simple TCL program
    tcl_code = "Ψ → Γ"
    
    # Parse and compile
    parser = TCLParser()
    compiler = TCLCompiler()
    
    expressions = parser.parse(tcl_code)
    compiled = compiler.compile(expressions)
    
    # Create execution context
    context = TCLExecutionContext()
    
    # Execute
    runtime = TCLRuntime()
    result = runtime.execute(compiled, context)
    
    print("TCL Runtime Demonstration")
    print("=" * 40)
    print(f"Code: {tcl_code}")
    print(f"Result: {result}")
    print(f"Enhancement Level: {result['metrics']['enhancement_level']:.2f}x")
    print(f"Cognitive Effects: {result['cognitive_effects']}")

if __name__ == "__main__":
    demonstrate_tcl_runtime()