#!/usr/bin/env python3
""" BASIC Interpreter """
#
# Multiple statements per line separated by COLON
#
###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################
# Token types
# EOF (end-of-file) token is used to indicate
# there is no more input left for lexical analysis
ID, ASSIGN, NUMBER, PLUS, MINUS, MUL, DIV, \
    LPAREN, RPAREN, LET, NEWLINE, EOF = (
        'ID', 'ASSIGN', 'NUMBER', 'PLUS', 'MINUS', 'MUL', 'DIV', 'LPAREN',
        'RPAREN', 'LET', 'NEWLINE', 'EOF')


class Token():
    """BASIC token"""

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance,

        Examples:
            Token(NUMBER, 3)
            Token(MUL, '*')
        """
        return f"Token({self.type}, {self.value})"

    def __repr__(self):
        return self.__str__()


class Lexer():
    """BASIC Lexical Analyser / Scanner"""

    RESERVED_KEYWORDS = {
        LET: Token(LET, LET),
    }

    def __init__(self, text):
        self.text = text  # client string input, eg. "3*5"
        self.pos = 0  # self.pos is an index into self.text
        self.current_char = self.text[self.pos]  # current character

    def error(self):
        raise Exception('Invalid Character')

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        else:
            return self.text[peek_pos]

    def advance(self):
        """Advance the 'pos' pointer and set the 'current_char' variable"""
        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        """Return a number consumed from the input"""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()

            while (self.current_char is not None
                   and self.current_char.isdigit()):
                result += self.current_char
                self.advance()

        return float(result)

    def _id(self):
        """Handle identifiers and reserved keywords"""
        result = ''
        while (self.current_char is not None
               and self.current_char.isidentifier()):
            result += self.current_char
            self.advance()

        token = self.RESERVED_KEYWORDS.get(result.upper(),
                                           Token(ID, result.lower()))
        return token

    def get_next_token(self):
        """Lexical analyser (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char == '\n':
                self.advance()
                return Token(NEWLINE, '\n')

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return Token(NUMBER, self.number())

            if self.current_char == '=':
                self.advance()
                return Token(ASSIGN, '=')

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            self.error()

        return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################


class AST():
    """Abstract Syntax Tree node"""
    pass


class Program(AST):
    """Program AST node"""

    def __init__(self):
        self.children = []


class Assign(AST):
    """Assign AST node"""

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Var(AST):
    """Variable AST node"""

    def __init__(self, token):
        self.token = token
        self.value = token.value


class NoOp(AST):
    """Empty AST node"""
    pass


class BinOp(AST):
    """Binary Operator AST node"""

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class UnaryOp():
    """Unary Operator AST node"""

    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Num(AST):
    """Number AST node"""

    def __init__(self, token):
        self.token = token
        self.value = token.value


class Parser():
    """BASIC parser"""

    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first tokan taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        """program : statement_list"""
        nodes = self.statement_list()

        root = Program()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        """
        statement_list : statement
                       | statement NEWLINE statement_list
        """
        node = self.statement()

        results = [node]

        while self.current_token.type == NEWLINE:
            self.eat(NEWLINE)
            results.append(self.statement())

        return results

    def statement(self):
        """
        statement : assignment_statement
                  | empty
        """
        if self.current_token.type == LET:
            self.eat(LET)
            node = self.assignment_statement()
        elif self.current_token.type == ID:
            node = self.assignment_statement()
        else:
            node = self.empty()
        return node

    def assignment_statement(self):
        """
        assignment_statement : variable ASSIGN expr
        """
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        """
        variable : ID
        """
        node = Var(self.current_token)
        self.eat(ID)
        return node

    def empty(self):
        return NoOp()

    def factor(self):
        """
        factor : PLUS factor
               | MINUS factor
               | NUMBER
               | LPAREN expr RPAREN
               | variable
        """
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == NUMBER:
            self.eat(NUMBER)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
        else:
            node = self.variable()
            return node

    def term(self):
        """term : factor ((MUL | DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (MUL, DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == DIV:
                self.eat(DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def expr(self):
        """Arithmetic expression parser

        expr   : term ((PLUS | MINUS) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : (PLUS | MINUS) NUMBER | LPAREN expr RPAREN | variable
        """
        node = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        node = self.program()
        if self.current_token.type != EOF:
            self.error()

        return node


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################


class NodeVisitor():
    """AST Node Visitor"""

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")


class Interpreter(NodeVisitor):
    """BASIC Interpreter"""

    GLOBAL_SCOPE = {}

    def __init__(self, parser):
        self.parser = parser

    def visit_Program(self, node):
        for child in node.children:
            self.visit(child)

    def visit_Assign(self, node):
        var_name = node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)

    def visit_Var(self, node):
        var_name = node.value
        val = self.GLOBAL_SCOPE.get(var_name)
        if val is None:
            raise NameError(repr(var_name))
        else:
            return val

    def visit_NoOp(self, node):
        pass

    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)

    def visit_Num(self, node):
        return node.value

    def interpret(self):
        tree = self.parser.parse()
        return self.visit(tree)


def main():
    with open('test.bas', 'r') as file:
        text = file.read()

        print(text + '\n')

        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        interpreter.interpret()
        print(interpreter.GLOBAL_SCOPE)


if __name__ == '__main__':
    main()
