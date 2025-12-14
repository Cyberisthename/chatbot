# Programming Concepts Guide

## Functions and Methods

Functions are fundamental building blocks in programming. They encapsulate reusable pieces of code that perform specific tasks. A function typically takes input parameters, processes them, and returns a result.

### Python Function Example

```python
def calculate_area(length, width):
    """Calculate the area of a rectangle."""
    return length * width

# Usage
area = calculate_area(5, 3)
print(f"The area is: {area}")
```

### Key Function Concepts

- **Parameters**: Input values that functions accept
- **Return values**: Output values that functions provide
- **Scope**: Variables defined inside functions are local
- **Recursion**: Functions calling themselves

## Object-Oriented Programming

OOP is a programming paradigm that organizes code around objects - entities that combine data and the functions that operate on that data.

### Classes and Objects

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
        return f"Deposited ${amount}. New balance: ${self.balance}"
    
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        else:
            return "Insufficient funds"

# Usage
account = BankAccount("Alice", 100)
print(account.deposit(50))
```

### OOP Principles

1. **Encapsulation**: Bundling data and methods together
2. **Inheritance**: Creating new classes from existing ones
3. **Polymorphism**: Different classes using the same interface
4. **Abstraction**: Hiding complex implementation details

## Error Handling

Robust programs need to handle errors gracefully. Python uses try-except blocks for error handling.

```python
def safe_divide(a, b):
    try:
        result = a / b
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except TypeError:
        return "Error: Invalid input types"
    finally:
        print("Division attempt completed")
```

## Best Practices

1. **Write clean, readable code**
2. **Use meaningful variable names**
3. **Comment complex logic**
4. **Test your code thoroughly**
5. **Follow PEP 8 style guide for Python**