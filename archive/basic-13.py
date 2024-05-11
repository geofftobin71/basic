#!/usr/bin/env python3
""" BASIC Interpreter """
import argparse
import sys
from enum import Enum

#
# Error Codes
# Token Enum
# Load BASIC file from command line
#
###############################################################################
#                                                                             #
#  Error Codes                                                                #
#                                                                             #
###############################################################################


class ErrorCode(Enum):
    UNEXPECTED_TOKEN = 'Unexpected token'
    ID_NOT_FOUND = 'Identifier not found'
    DUPLICATE_ID = 'Duplicate id found'


class Error(Exception):

    def __init__(self, error_code=None, token=None, message=None):
        super().__init__(message)
        self.error_code = error_code
        self.token = token


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class SemanticError(Error):
    pass


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################
# Token types
# EOF (end-of-file) token is used to indicate
# there is no more input left for lexical analysis


class TokenType(Enum):
    # single character token types
    PLUS = '+'
    MINUS = '-'
    MUL = '*'
    DIV = '/'
    LPAREN = '('
    RPAREN = ')'
    COLON = ':'
    NEWLINE = '\n'
    # reserved keywords
    LET = 'LET'
    # misc
    LINENUM = 'LINENUM'
    ID = 'ID'
    ASSIGN = '='
    NUMBER = 'NUMBER'
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
            Token(NUMBER, 3)
            Token(MUL, '*')
        """
        return 'Token({type}, {value}, position={lineno}:{column})'.format(
            type=self.type,
            value=repr(self.value),
            lineno=self.lineno,
            column=self.column)

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
# print(RESERVED_KEYWORDS)


class Lexer():
    """ BASIC Lexical Analyser / Scanner """

    def __init__(self, text):
        self.text = text  # client string input, eg. "3*5"
        self.pos = 0  # self.pos is an index into self.text
        self.current_char = self.text[self.pos]  # current character
        self.newline = True
        self.lineno = 1
        self.column = 1

    def error(self):
        s = "Lexer error on '{lexeme}' line: {lineno} column: {column}".format(
            lexeme=self.current_char, lineno=self.lineno, column=self.column)

        raise LexerError(message=s)

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        else:
            return self.text[peek_pos]

    def advance(self):
        """ Advance the 'pos' pointer and set the 'current_char' variable """
        if self.current_char == '\n':
            self.lineno += 1
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
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        return int(result)

    def number(self):
        """ Return a number consumed from the input """
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
        """ Handle identifiers and reserved keywords """

        # create a new token with current line and column number
        token = Token(type=None,
                      value=None,
                      lineno=self.lineno,
                      column=self.column)

        value = ''
        while (self.current_char is not None and self.current_char.isalnum()):
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

        return token

    def get_next_token(self):
        """ Lexical analyser (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:
            if self.current_char == '\n':
                self.advance()
                self.newline = True
                return Token(type=TokenType.NEWLINE,
                             value=TokenType.NEWLINE.value,
                             lineno=self.lineno,
                             column=self.column)

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                if self.newline:
                    self.newline = False
                    return Token(type=TokenType.LINENUM,
                                 value=self.line_number(),
                                 lineno=self.lineno,
                                 column=self.column)
                else:
                    return Token(type=TokenType.NUMBER,
                                 value=self.number(),
                                 lineno=self.lineno,
                                 column=self.column)

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
            """
            if self.current_char == ':':
                self.advance()
                return Token(TokenType.COLON, ':')

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
            """

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


class Program(AST):
    """ Program AST node """

    def __init__(self):
        self.line_numbers = {}
        self.children = []


class Assign(AST):
    """ Assign AST node """

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Var(AST):
    """ Variable AST node """

    def __init__(self, token):
        self.token = token
        self.value = token.value


class NoOp(AST):
    """ Empty AST node """
    pass


class BinOp(AST):
    """ Binary Operator AST node """

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class UnaryOp():
    """ Unary Operator AST node """

    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Num(AST):
    """ Number AST node """

    def __init__(self, token):
        self.token = token
        self.value = token.value


class Parser():
    """ BASIC parser """

    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first tokan taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self, error_code, token):
        raise ParserError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}'
        )

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token
            )

    def program(self):
        """ program : (line)* """
        root = Program()

        while self.current_token.type != TokenType.EOF:
            line_number, nodes = self.line()

            root.line_numbers[line_number] = len(root.children)
            for node in nodes:
                root.children.append(node)

        return root

    def line(self):
        """ line : LINENUM statement_list NEWLINE """
        line_number = self.current_token.value
        self.eat(TokenType.LINENUM)
        nodes = self.statement_list()
        self.eat(TokenType.NEWLINE)

        return line_number, nodes

    def statement_list(self):
        """
        statement_list : statement
                       | statement COLON statement_list
        """
        node = self.statement()

        results = [node]

        while self.current_token.type == TokenType.COLON:
            self.eat(TokenType.COLON)
            results.append(self.statement())

        return results

    def statement(self):
        """
        statement : assignment_statement
                  | empty
        """
        if self.current_token.type == TokenType.LET:
            self.eat(TokenType.LET)
            node = self.assignment_statement()
        elif self.current_token.type == TokenType.ID:
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
        self.eat(TokenType.ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        """
        variable : ID
        """
        node = Var(self.current_token)
        self.eat(TokenType.ID)
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
        if token.type == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return Num(token)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
        else:
            node = self.variable()
            return node

    def term(self):
        """ term : factor ((MUL | DIV) factor)* """
        node = self.factor()

        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            token = self.current_token
            if token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
            elif token.type == TokenType.DIV:
                self.eat(TokenType.DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def expr(self):
        """ Arithmetic expression parser

        expr   : term ((PLUS | MINUS) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : (PLUS | MINUS) NUMBER | LPAREN expr RPAREN | variable
        """
        node = self.term()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        node = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error()

        return node


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
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


class Interpreter(NodeVisitor):
    """ BASIC Interpreter """

    def __init__(self, tree):
        self.tree = tree
        self.GLOBAL_SCOPE = {}

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
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == TokenType.PLUS:
            return +self.visit(node.expr)
        elif op == TokenType.MINUS:
            return -self.visit(node.expr)

    def visit_Num(self, node):
        return node.value

    def interpret(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)


def main():
    parser = argparse.ArgumentParser(
        description='BASIC Interpreter'
    )
    parser.add_argument('filename', help='BASIC source file')

    args = parser.parse_args()

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
        interpreter.interpret()
        print(interpreter.GLOBAL_SCOPE)


if __name__ == '__main__':
    main()
