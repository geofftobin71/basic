#!/usr/bin/env python3
""" BASIC Interpreter """
import argparse
import sys
from enum import Enum

#
# Code Refactor
# Refector expressions to include exponent, logical and comparison expressions
#


class Error(Exception):

    def __init__(self, message, lineno=None, column=None):
        self.message = message
        if lineno and column:
            self.message += f'. line: {lineno} column: {column}'
        super().__init__(self.message)


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class RuntimeError(Error):
    pass


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################


class TokenType(Enum):
    # single character token types
    PLUS = '+'
    MINUS = '-'
    MUL = '*'
    DIV = '/'
    POWER = '^'
    EQUAL = '='
    GREATER = '>'
    LESS = '<'
    LPAREN = '('
    RPAREN = ')'
    COLON = ':'
    SEMICOLON = ';'
    COMMA = ','
    NEWLINE = '\n'  # must precede reserved keywords
    # reserved keywords
    LET = 'LET'
    REM = 'REM'
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'
    GOTO = 'GOTO'
    GOSUB = 'GOSUB'
    RETURN = 'RETURN'
    STOP = 'STOP'
    END = 'END'
    # misc
    LINENUM = 'LINENUM'  # must follow reserved keywords
    ID = 'ID'
    ASSIGN = '='
    NOTEQUAL = '<>'
    GEQUAL = '>='
    LEQUAL = '<='
    NUMBER = 'NUMBER'
    STRING = 'STRING'
    EOF = 'EOF'


class Token():
    """ BASIC token """

    def __init__(self, type, value, lineno=None, column=None):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __str__(self):
        """ String representation of the class instance,

        Examples:
            Token(NUMBER, 3, line: 1 column: 2)
            Token(MUL, '*', line: 3 column: 4)
        """
        s = f'Token({self.type}, {repr(self.value)}, '
        s += f'line: {self.lineno} column: {self.column})'

        return s

    def __repr__(self):
        return self.__str__()


def _build_reserved_keywords():
    """ Build a dictionary of reserved keywords

    The function relies on the fact that in the TokenType
    enumeration the beginning of the block of reserved keywords starts
    after NEWLINE and ends before LINENUM
    """
    tt_list = list(TokenType)
    start_index = tt_list.index(TokenType.NEWLINE) + 1
    end_index = tt_list.index(TokenType.LINENUM)
    reserved_keywords = {
        token_type.value: token_type
        for token_type in tt_list[start_index:end_index]
    }
    return reserved_keywords


RESERVED_KEYWORDS = _build_reserved_keywords()


class Lexer():
    """ BASIC Lexical Analyser / Scanner """

    def __init__(self, text):
        self.text = text  # BASIC program source
        self.pos = 0  # self.pos is an index into self.text
        self.current_char = self.text[self.pos]  # current character
        self.newline = True  # newline = True -> digits = Line Number
        self.lineno = None  # current BASIC line number
        self.column = 1  # current column

    def error(self):
        message = f"Unexpected character '{self.current_char}'"
        raise LexerError(message, self.lineno, self.column)

    def peek(self):
        """ peek at next character without advancing pos pointer """
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        else:
            return self.text[peek_pos]

    def advance(self):
        """ Advance the 'pos' pointer and set the 'current_char' variable """
        if self.current_char == '\n':
            self.column = 0

        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]
            self.column += 1

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def line_number(self):
        """ Return a line number consumed from the input """

        column = self.column

        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        self.lineno = int(result)

        # create a new token with current line and column number
        token = Token(type=TokenType.LINENUM,
                      value=self.lineno,
                      lineno=self.lineno,
                      column=column)

        return token

    def number(self):
        """ Return a number consumed from the input """

        # create a new token with current line and column number
        token = Token(type=TokenType.NUMBER,
                      value=None,
                      lineno=self.lineno,
                      column=self.column)

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

        token.value = float(result)
        return token

    def _id(self):
        """ Handle identifiers and reserved keywords """

        # create a new token with current line and column number
        token = Token(type=None,
                      value=None,
                      lineno=self.lineno,
                      column=self.column)

        value = ''
        while (self.current_char is not None
               and (self.current_char.isalnum() or self.current_char == '$'
                    or self.current_char == '%')):
            value += self.current_char
            self.advance()

        token_type = RESERVED_KEYWORDS.get(value.upper())
        if token_type is None:
            token.type = TokenType.ID
            token.value = value.lower()
        else:
            # reserved keyword
            token.type = token_type
            token.value = value.upper()
            if token.type == TokenType.REM:
                while self.current_char != '\n':
                    self.advance()
            elif token.type == TokenType.STOP:
                token.type = TokenType.END
                token.value = TokenType.END.value

        return token

    def get_next_token(self):
        """ Lexical analyser (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:
            if self.current_char == "'":  # single quote
                token = Token(type=TokenType.REM,
                              value=TokenType.REM.value,
                              lineno=self.lineno,
                              column=self.column)
                while self.current_char != '\n':
                    self.advance()
                return token

            if self.current_char == '>' and self.peek() == '=':
                token = Token(type=TokenType.GEQUAL,
                              value=TokenType.GEQUAL.value,
                              lineno=self.lineno,
                              column=self.column)
                self.advance()
                self.advance()
                return token

            if self.current_char == '<' and self.peek() == '=':
                token = Token(type=TokenType.LEQUAL,
                              value=TokenType.LEQUAL.value,
                              lineno=self.lineno,
                              column=self.column)
                self.advance()
                self.advance()
                return token

            if self.current_char == '<' and self.peek() == '>':
                token = Token(type=TokenType.NOTEQUAL,
                              value=TokenType.NOTEQUAL.value,
                              lineno=self.lineno,
                              column=self.column)
                self.advance()
                self.advance()
                return token

            if self.current_char == '>' and self.peek() == '<':
                token = Token(type=TokenType.NOTEQUAL,
                              value=TokenType.NOTEQUAL.value,
                              lineno=self.lineno,
                              column=self.column)
                self.advance()
                self.advance()
                return token

            if self.current_char == '\n':
                token = Token(type=TokenType.NEWLINE,
                              value=TokenType.NEWLINE.value,
                              lineno=self.lineno,
                              column=self.column)
                self.advance()
                self.newline = True
                return token

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                if self.newline:
                    self.newline = False
                    return self.line_number()
                else:
                    return self.number()

            # single character tokens
            try:
                # get enum member by value eg.
                # TokenType(':') --> TokenType.COLON
                token_type = TokenType(self.current_char)
            except ValueError:
                # no enum member with value equal to self.current_char
                self.error()
            else:
                # create a token with a single-character lexeme as its value
                token = Token(type=token_type,
                              value=token_type.value,
                              lineno=self.lineno,
                              column=self.column)
                self.advance()
                return token

            self.error()

        return Token(TokenType.EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################


class AST():
    """ Abstract Syntax Tree node """
    pass


class ProgramNode(AST):

    def __init__(self):
        self.line_numbers = {}
        self.children = []


class AssignNode(AST):

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class VarNode(AST):

    def __init__(self, token):
        self.token = token
        self.value = token.value


class NoOpNode(AST):
    pass


class BinOpNode(AST):

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class UnaryOpNode(AST):

    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class NumNode(AST):

    def __init__(self, token):
        self.token = token
        self.value = token.value


class GoToNode(AST):

    def __init__(self, expr):
        self.expr = expr


class GoSubNode(AST):

    def __init__(self, expr):
        self.expr = expr


class ReturnNode(AST):

    def __init__(self, token):
        self.token = token


class EndNode(AST):
    pass


class Parser():
    """ BASIC parser """

    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first tokan taken from the input
        self.current_token = self.lexer.get_next_token()

    def _eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            token = self.current_token
            message = f"Unexpected Token '{token.value}'"
            raise ParserError(message, token.lineno, token.column)

    def parse_program(self):
        """ program : (line)* """
        root = ProgramNode()

        while self.current_token.type != TokenType.EOF:
            line_number, nodes = self.parse_line()

            root.line_numbers[line_number] = len(root.children)
            for node in nodes:
                root.children.append(node)

        return root

    def parse_line(self):
        """ line : LINENUM statement_list NEWLINE """
        line_number = self.current_token.value
        self._eat(TokenType.LINENUM)
        nodes = self.parse_statement_list()
        self._eat(TokenType.NEWLINE)

        return line_number, nodes

    def parse_statement_list(self):
        """
        statement_list : statement
                       | statement COLON statement_list
        """
        node = self.parse_statement()

        results = [node]

        while self.current_token.type == TokenType.COLON:
            self._eat(TokenType.COLON)
            results.append(self.parse_statement())

        return results

    def parse_statement(self):
        """
        statement : assignment_statement
                  | goto_statement
                  | gosub_statement
                  | return_statement
                  | end_statement
                  | empty
        """
        if self.current_token.type == TokenType.REM:
            node = self.parse_rem_statement()
        elif self.current_token.type == TokenType.GOTO:
            node = self.parse_goto_statement()
        elif self.current_token.type == TokenType.GOSUB:
            node = self.parse_gosub_statement()
        elif self.current_token.type == TokenType.RETURN:
            node = self.parse_return_statement()
        elif self.current_token.type == TokenType.END:
            node = self.parse_end_statement()
        elif self.current_token.type == TokenType.LET:
            self._eat(TokenType.LET)
            node = self.parse_assignment_statement()
        elif self.current_token.type == TokenType.ID:
            node = self.parse_assignment_statement()
        else:
            node = self.parse_empty()

        return node

    def parse_assignment_statement(self):
        """
        assignment_statement : variable ASSIGN expr
        """
        left = self.parse_variable()
        token = self.current_token
        self._eat(TokenType.ASSIGN)
        right = self.parse_expr()
        node = AssignNode(left, token, right)
        return node

    def parse_rem_statement(self):
        self._eat(TokenType.REM)
        return NoOpNode()

    def parse_goto_statement(self):
        """ goto_statement : GOTO LINENUM  """
        self._eat(TokenType.GOTO)
        return GoToNode(self.parse_expr())

    def parse_gosub_statement(self):
        """ gosub_statement : GOSUB LINENUM  """
        self._eat(TokenType.GOSUB)
        return GoSubNode(self.parse_expr())

    def parse_return_statement(self):
        """ return_statement : RETURN  """
        node = ReturnNode(self.current_token)
        self._eat(TokenType.RETURN)
        return node

    def parse_end_statement(self):
        """ end_statement : END """
        self._eat(TokenType.END)
        return EndNode()

    def parse_variable(self):
        """
        variable : ID
        """
        node = VarNode(self.current_token)
        self._eat(TokenType.ID)
        return node

    def parse_empty(self):
        return NoOpNode()

    def parse_value(self):
        """
        value : NUMBER
              | LPAREN expr RPAREN
              | variable
        """
        token = self.current_token
        if token.type == TokenType.NUMBER:
            self._eat(TokenType.NUMBER)
            node = NumNode(token)
        elif token.type == TokenType.LPAREN:
            self._eat(TokenType.LPAREN)
            node = self.parse_expr()
            self._eat(TokenType.RPAREN)
        else:
            node = self.parse_variable()

        return node

    def parse_power_expr(self):
        """ power_expr : value (POWER value)* """
        node = self.parse_value()

        while self.current_token.type == TokenType.POWER:
            token = self.current_token
            self._eat(TokenType.POWER)

            node = BinOpNode(left=node, op=token, right=self.parse_value())

        return node

    def parse_unary_expr(self):
        """
        unary_expr : PLUS power_expr
                   | MINUS power_expr
                   | power_expr
        """
        token = self.current_token
        if token.type == TokenType.PLUS:
            self._eat(TokenType.PLUS)
            node = UnaryOpNode(token, self.parse_power_expr())
        elif token.type == TokenType.MINUS:
            self._eat(TokenType.MINUS)
            node = UnaryOpNode(token, self.parse_power_expr())
        else:
            node = self.parse_power_expr()

        return node

    def parse_mult_expr(self):
        """ mult_expr : unary_expr ((MUL | DIV) unary_expr)* """
        node = self.parse_unary_expr()

        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            token = self.current_token
            self._eat(token.type)

            node = BinOpNode(left=node,
                             op=token,
                             right=self.parse_unary_expr())

        return node

    def parse_add_expr(self):
        """ add_expr : mult_expr ((PLUS | MINUS) mult_expr)* """
        node = self.parse_mult_expr()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            self._eat(token.type)

            node = BinOpNode(left=node, op=token, right=self.parse_mult_expr())

        return node

    def parse_compare_expr(self):
        """
        compare_expr : add_expr ((EQUAL | NOTEQUAL | GREATER | LESS |
                                  GEQUAL | LEQUAL) add_expr)*
        """
        node = self.parse_add_expr()

        while self.current_token.type in (TokenType.EQUAL, TokenType.NOTEQUAL,
                                          TokenType.GREATER, TokenType.LESS,
                                          TokenType.GEQUAL, TokenType.LEQUAL):
            token = self.current_token
            self._eat(token.type)

            node = BinOpNode(left=node, op=token, right=self.parse_add_expr())

        return node

    def parse_not_expr(self):
        """ not_expr : (NOT)* compare_expr """
        token = self.current_token
        if token.type == TokenType.NOT:
            self._eat(TokenType.NOT)
            node = UnaryOpNode(token, self.parse_compare_expr())
        else:
            node = self.parse_compare_expr()

        return node

    def parse_and_expr(self):
        """ and_expr : not_expr (AND not_expr)* """
        node = self.parse_not_expr()

        while self.current_token.type == TokenType.AND:
            token = self.current_token
            self._eat(token.type)

            node = BinOpNode(left=node, op=token, right=self.parse_not_expr())

        return node

    def parse_expr(self):
        """ Expression Parser

        expr         : and_expr (OR and_expr)*
        and_expr     : not_expr (AND not_expr)*
        not_expr     : (NOT)* compare_expr
        compare_expr : add_expr ((EQUAL | NOTEQUAL | GREATER | LESS |
                                  GEQUAL | LEQUAL) add_expr)*
        add_expr     : mult_expr ((PLUS | MINUS) mult_expr)*
        mult_expr    : unary_expr ((MUL | DIV) unary_expr)*
        unary_expr   : (PLUS | MINUS)* power_expr
        power_expr   : value (POWER value)*
        value        : NUMBER | LPAREN expr RPAREN | variable
        """
        node = self.parse_and_expr()

        while self.current_token.type == TokenType.OR:
            token = self.current_token
            self._eat(token.type)

            node = BinOpNode(left=node, op=token, right=self.parse_and_expr())

        return node

    def parse(self):
        node = self.parse_program()
        if self.current_token.type != TokenType.EOF:
            raise ParserError("EOF not found")

        return node


###############################################################################
#                                                                             #
#  AST NODE VISITOR                                                           #
#                                                                             #
###############################################################################


class NodeVisitor():
    """ AST Node Visitor """

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f'No visit_{type(node).__name__} method')


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################


class Interpreter(NodeVisitor):
    """ BASIC Interpreter """

    def __init__(self, tree):
        self.tree = tree
        self.next_node = 0
        self.return_stack = []
        self.line_numbers = {}
        self.GLOBAL_SCOPE = {}

    def visit_ProgramNode(self, node):
        self.line_numbers = node.line_numbers

        current_node = 0
        while True:
            self.next_node = current_node + 1
            self.visit(node.children[current_node])
            current_node = self.next_node
            if current_node is None or current_node >= len(node.children):
                break

    def visit_AssignNode(self, node):
        var_name = node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)

    def visit_VarNode(self, node):
        var_name = node.value
        val = self.GLOBAL_SCOPE.get(var_name)
        if val is None:
            message = f"Identifier '{node.token.value}' not found"
            raise RuntimeError(message, node.token.lineno, node.token.column)
        else:
            return val

    def visit_NoOpNode(self, node):
        pass

    def visit_BinOpNode(self, node):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.DIV:
            return self.visit(node.left) / self.visit(node.right)
        elif node.op.type == TokenType.POWER:
            return self.visit(node.left)**self.visit(node.right)
        elif node.op.type == TokenType.EQUAL:
            return 1 if self.visit(node.left) == self.visit(node.right) else 0
        elif node.op.type == TokenType.NOTEQUAL:
            return 1 if self.visit(node.left) != self.visit(node.right) else 0
        elif node.op.type == TokenType.GREATER:
            return 1 if self.visit(node.left) > self.visit(node.right) else 0
        elif node.op.type == TokenType.LESS:
            return 1 if self.visit(node.left) < self.visit(node.right) else 0
        elif node.op.type == TokenType.GEQUAL:
            return 1 if self.visit(node.left) >= self.visit(node.right) else 0
        elif node.op.type == TokenType.LEQUAL:
            return 1 if self.visit(node.left) <= self.visit(node.right) else 0
        elif node.op.type == TokenType.AND:
            return 1 if (self.visit(node.left) != 0) and \
                        (self.visit(node.right) != 0) else 0
        elif node.op.type == TokenType.OR:
            return 1 if (self.visit(node.left) != 0) or \
                        (self.visit(node.right) != 0) else 0

    def visit_UnaryOpNode(self, node):
        op = node.op.type
        if op == TokenType.PLUS:
            return +self.visit(node.expr)
        elif op == TokenType.MINUS:
            return -self.visit(node.expr)
        elif op == TokenType.NOT:
            return 1 if not (self.visit(node.expr) != 0) else 0

    def visit_NumNode(self, node):
        return node.value

    def visit_GoToNode(self, node):
        self.next_node = self.line_numbers.get(int(self.visit(node.expr)))
        if self.next_node is None:
            message = f"Line {int(node.expr.value)} not found"
            lineno = node.expr.token.lineno
            column = node.expr.token.column
            raise RuntimeError(message, lineno, column)

    def visit_GoSubNode(self, node):
        self.return_stack.append(self.next_node)
        self.next_node = self.line_numbers.get(int(self.visit(node.expr)))
        if self.next_node is None:
            message = f"Line {int(node.expr.value)} not found"
            lineno = node.expr.token.lineno
            column = node.expr.token.column
            raise RuntimeError(message, lineno, column)

    def visit_ReturnNode(self, node):
        try:
            self.next_node = self.return_stack.pop()
        except IndexError:
            message = "Return without Gosub"
            lineno = node.token.lineno
            column = node.token.column
            raise RuntimeError(message, lineno, column)

    def visit_EndNode(self, node):
        self.next_node = None

    def run(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)


def main():
    arg_parser = argparse.ArgumentParser(description='BASIC Interpreter')
    arg_parser.add_argument('filename', help='BASIC source file')

    args = arg_parser.parse_args()

    with open(args.filename, 'r') as file:
        text = file.read()

        lexer = Lexer(text)
        try:
            parser = Parser(lexer)
            tree = parser.parse()
        except (LexerError, ParserError) as e:
            print(e.message)
            sys.exit(1)

        interpreter = Interpreter(tree)
        try:
            interpreter.run()
        except RuntimeError as e:
            print(e.message)
            sys.exit(1)

        print(interpreter.GLOBAL_SCOPE)


if __name__ == '__main__':
    main()
