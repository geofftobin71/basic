#!/usr/bin/env python3
""" BASIC Interpreter """
import argparse
import math
import os
import random
import select
import sys
import termios
from enum import Enum

###############################################################################
#                                                                             #
#  GLOBAL FUNCTIONS                                                           #
#                                                                             #
###############################################################################


def getkey():
    old_settings = termios.tcgetattr(sys.stdin)
    new_settings = termios.tcgetattr(sys.stdin)
    new_settings[3] &= ~(termios.ECHO | termios.ICANON)
    new_settings[6][termios.VMIN] = 0
    new_settings[6][termios.VTIME] = 0
    termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, new_settings)
    try:
        if select.select([sys.stdin], [], [], 0.01) == ([sys.stdin], [], []):
            b = sys.stdin.read(1)
            return b[0].upper()
        else:
            return ""
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_settings)


def init_list(dims, val):
    if len(dims) == 0:
        raise ValueError("Requires at least 1 dimension")

    if len(dims) == 1:
        return [val for _ in range(dims[0])]

    return [init_list(dims[1:], val=val) for _ in range(dims[0])]


def set_list(data, value, idxs, tokens):
    if len(idxs) == 0:
        raise ValueError("Requires at least 1 dimension")

    if idxs[0] < 0 or idxs[0] >= len(data):
        out_of_range_error(tokens[0])

    if len(idxs) == 1:
        data[idxs[0]] = value
    else:
        set_list(data[idxs[0]], value, idxs[1:], tokens[1:])


def get_list(data, idxs, tokens):
    if len(idxs) == 0:
        raise ValueError("Requires at least 1 dimension")

    if idxs[0] < 0 or idxs[0] >= len(data):
        out_of_range_error(tokens[0])

    if len(idxs) == 1:
        return data[idxs[0]]
    else:
        return get_list(data[idxs[0]], idxs[1:], tokens[1:])


def get_dims(data):
    if not isinstance(data, list):
        return 0
    return get_dims(data[0]) + 1


###############################################################################
#                                                                             #
#  ERRORS                                                                     #
#                                                                             #
###############################################################################


class Error(Exception):

    def __init__(self, message, lineno=None, column=None):
        self.message = message
        if lineno and column:
            self.message += f". line: {lineno} column: {column}"
        super().__init__(self.message)


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class RuntimeError(Error):
    pass


def unexpected_token_error(token):
    if type(token.value) is str:
        message = f"Unexpected Token '{token.value}'"
    else:
        message = f"Unexpected Token '{token.value:g}'"
    raise ParserError(message, token.lineno, token.column)


def type_mismatch_error(token):
    if type(token.value) is str:
        message = f"Type mismatch at '{token.value}'"
    else:
        message = f"Type mismatch at '{token.value:g}'"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def out_of_data_error(token):
    message = f"Out of data at '{token.value}'"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def out_of_range_error(token):
    if type(token.value) is str:
        message = f"Out of range at '{token.value}'"
    else:
        message = f"Out of range at '{token.value:g}'"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def identifier_not_found_error(token):
    message = f"Identifier '{token.value}' not found"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def empty_string_error(token):
    message = f"Empty string at '{token.value}'"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def line_not_found_error(token):
    message = f"Line {int(token.value)} not found"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def return_without_gosub_error(token):
    message = "Return without Gosub"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def next_without_for_error(token):
    message = "Next without For"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


def redimd_array_error(token):
    message = "ReDim'd array  at '{token.value}'"
    lineno = token.lineno
    column = token.column
    raise RuntimeError(message, lineno, column)


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################


class TokenType(Enum):
    # single character token types
    PLUS = "+"
    MINUS = "-"
    MUL = "*"
    DIV = "/"
    POWER = "^"
    EQUAL = "="
    GREATER = ">"
    LESS = "<"
    LPAREN = "("
    RPAREN = ")"
    COLON = ":"
    SEMICOLON = ";"
    COMMA = ","
    NEWLINE = "\n"  # must precede reserved keywords
    # reserved keywords
    AND = "AND"
    ABS = "ABS"
    ASC = "ASC"
    ATN = "ATN"
    CHR = "CHR$"
    CLS = "CLS"
    COS = "COS"
    DATA = "DATA"
    DIM = "DIM"
    ELSE = "ELSE"
    END = "END"
    EXP = "EXP"
    FOR = "FOR"
    GET = "GET"
    GOTO = "GOTO"
    GOSUB = "GOSUB"
    IF = "IF"
    INPUT = "INPUT"
    INT = "INT"
    LEFT = "LEFT$"
    LEN = "LEN"
    LET = "LET"
    LOG = "LOG"
    MID = "MID$"
    NEXT = "NEXT"
    NOT = "NOT"
    ON = "ON"
    OR = "OR"
    PRINT = "PRINT"
    READ = "READ"
    REM = "REM"
    RESTORE = "RESTORE"
    RETURN = "RETURN"
    RIGHT = "RIGHT$"
    RND = "RND"
    SGN = "SGN"
    SIN = "SIN"
    SPC = "SPC"
    SQR = "SQR"
    STR = "STR$"
    TAB = "TAB"
    TAN = "TAN"
    THEN = "THEN"
    TO = "TO"
    STEP = "STEP"
    STOP = "STOP"
    VAL = "VAL"
    # misc
    LINENUM = "LINENUM"  # must follow reserved keywords
    ID = "ID"
    ASSIGN = "="
    NOTEQUAL = "<>"
    GEQUAL = ">="
    LEQUAL = "<="
    NUMBER = "NUMBER"
    STRING = "STRING"
    EOF = "EOF"


class Token():
    """ BASIC token """

    def __init__(self, type, value, lineno=None, column=None):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __str__(self):
        """
        String representation of the class instance,

        Examples:
            Token(NUMBER, 3, line: 1 column: 2)
            Token(MUL, "*", line: 3 column: 4)
        """
        s = f"Token({self.type}, {repr(self.value)}, "
        s += f"line: {self.lineno} column: {self.column})"

        return s

    def __repr__(self):
        return self.__str__()


def _build_reserved_keywords():
    """
    Build a dictionary of reserved keywords

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

        result = ""
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

        result = ""
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

    def string(self):
        """ Return a string consumed from the input """

        # create a new token with current line and column number
        token = Token(type=TokenType.STRING,
                      value=None,
                      lineno=self.lineno,
                      column=self.column)

        self.advance()  # eat the opening quote

        result = ""
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()

        self.advance()  # eat the closing quote

        token.value = result
        return token

    def _id(self):
        """ Handle identifiers and reserved keywords """

        # create a new token with current line and column number
        token = Token(type=None,
                      value=None,
                      lineno=self.lineno,
                      column=self.column)

        value = ""
        while (self.current_char is not None
               and (self.current_char.isalnum() or self.current_char == '$'
                    or self.current_char == '%')):
            value += self.current_char
            self.advance()

        token_type = RESERVED_KEYWORDS.get(value.upper())
        if token_type is None:
            token.type = TokenType.ID
            token.value = value  # .lower()
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
        """
        Lexical analyser (also known as scanner or tokenizer)

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

            if self.current_char == '"':
                return self.string()

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
        self.nodes = []


class PrintNode(AST):

    def __init__(self):
        self.end = '\n'
        self.nodes = []


class AssignNode(AST):

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class VarNode(AST):

    def __init__(self, token):
        self.token = token
        self.value = token.value
        self.idxs = []


class NoOpNode(AST):
    pass


class BinOpNode(AST):

    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class LeftDollarNode(AST):

    def __init__(self, string, length):
        self.string = string
        self.length = length


class MidDollarNode(AST):

    def __init__(self, string, start, length):
        self.string = string
        self.start = start
        self.length = length


class RightDollarNode(AST):

    def __init__(self, string, length):
        self.string = string
        self.length = length


class UnaryOpNode(AST):

    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class NumNode(AST):

    def __init__(self, token):
        self.token = token
        self.value = token.value


class StringNode(AST):

    def __init__(self, token):
        self.token = token
        self.value = token.value


class IfNode(AST):

    def __init__(self, expr, skip_then):
        self.expr = expr
        self.skip_then = skip_then


class SkipNode(AST):

    def __init__(self, skip):
        self.skip = skip


class OnNode(AST):

    def __init__(self, expr, token):
        self.expr = expr
        self.token = token
        self.linenos = []


class GoToNode(AST):

    def __init__(self, expr):
        self.expr = expr


class GoSubNode(AST):

    def __init__(self, expr):
        self.expr = expr


class ReturnNode(AST):

    def __init__(self, token):
        self.token = token


class ForNode(AST):

    def __init__(self, assign, var, limit, step):
        self.assign = assign
        self.var = var
        self.limit = limit
        self.step = step


class NextNode(AST):

    def __init__(self, token):
        self.token = token


class InputNode(AST):

    def __init__(self):
        self.prompt = ""
        self.ids = []


class ReadNode(AST):

    def __init__(self):
        self.ids = []


class RestoreNode(AST):
    pass


class ClsNode(AST):
    pass


class DimNode(AST):

    def __init__(self, var):
        self.var = var
        self.dims = []


class GetNode(AST):

    def __init__(self, var):
        self.var = var


class EndNode(AST):
    pass


class Parser():
    """ BASIC parser """

    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first tokan taken from the input
        self.current_token = self.lexer.get_next_token()
        self.global_data = []

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            unexpected_token_error(self.current_token)

    def parse_program(self):
        """
        program : (line)*
        """
        root = ProgramNode()

        while self.current_token.type != TokenType.EOF:
            line_number, nodes = self.parse_line()

            root.line_numbers[line_number] = len(root.nodes)
            for node in nodes:
                root.nodes.append(node)

        return root

    def parse_line(self):
        """
        line : LINENUM statement_list NEWLINE
        """
        line_number = self.current_token.value
        self.eat(TokenType.LINENUM)
        nodes = self.parse_statement_list()
        self.eat(TokenType.NEWLINE)

        return line_number, nodes

    def parse_statement_list(self):
        """
        statement_list : statement
                       | statement COLON statement_list
        """
        node, nodes = self.parse_statement()

        results = [node]
        if nodes:
            results += nodes

        while self.current_token.type == TokenType.COLON:
            self.eat(TokenType.COLON)
            node, nodes = self.parse_statement()
            results.append(node)
            if nodes:
                results += nodes

        return results

    def parse_statement(self):
        """
        statement : assignment_statement
                  | rem_statement
                  | cls_statement
                  | print_statement
                  | input_statement
                  | get_statement
                  | on_statement
                  | goto_statement
                  | gosub_statement
                  | return_statement
                  | if_statement
                  | for_statement
                  | next_statement
                  | dim_statement
                  | data_statement
                  | read_statement
                  | restore_statement
                  | end_statement
                  | empty
        """
        nodes = None

        if self.current_token.type == TokenType.ID:
            node = self.parse_assignment_statement()
        elif self.current_token.type == TokenType.LET:
            self.eat(TokenType.LET)
            node = self.parse_assignment_statement()
        elif self.current_token.type == TokenType.REM:
            node = self.parse_rem_statement()
        elif self.current_token.type == TokenType.CLS:
            node = self.parse_cls_statement()
        elif self.current_token.type == TokenType.INPUT:
            node = self.parse_input_statement()
        elif self.current_token.type == TokenType.PRINT:
            node = self.parse_print_statement()
        elif self.current_token.type == TokenType.GET:
            node = self.parse_get_statement()
        elif self.current_token.type == TokenType.ON:
            node = self.parse_on_statement()
        elif self.current_token.type == TokenType.GOTO:
            node = self.parse_goto_statement()
        elif self.current_token.type == TokenType.GOSUB:
            node = self.parse_gosub_statement()
        elif self.current_token.type == TokenType.RETURN:
            node = self.parse_return_statement()
        elif self.current_token.type == TokenType.IF:
            node, nodes = self.parse_if_statement()
        elif self.current_token.type == TokenType.FOR:
            node = self.parse_for_statement()
        elif self.current_token.type == TokenType.NEXT:
            node, nodes = self.parse_next_statement()
        elif self.current_token.type == TokenType.DIM:
            node = self.parse_dim_statement()
        elif self.current_token.type == TokenType.READ:
            node = self.parse_read_statement()
        elif self.current_token.type == TokenType.DATA:
            node = self.parse_data_statement()
        elif self.current_token.type == TokenType.RESTORE:
            node = self.parse_restore_statement()
        elif self.current_token.type == TokenType.END:
            node = self.parse_end_statement()
        else:
            node = self.parse_empty()

        return node, nodes

    def parse_assignment_statement(self):
        """
        assignment_statement : variable ASSIGN expr
        """
        left = self.parse_variable()
        op = self.current_token
        self.eat(TokenType.ASSIGN)
        right = self.parse_expr()

        node = AssignNode(left, op, right)
        return node

    def parse_rem_statement(self):
        self.eat(TokenType.REM)
        return NoOpNode()

    def parse_cls_statement(self):
        self.eat(TokenType.CLS)
        return ClsNode()

    def parse_print_statement(self):
        """
        print_statement : PRINT expr (SEMICOLON expr)* (SEMICOLON)*
        """
        self.eat(TokenType.PRINT)
        node = PrintNode()

        while (self.current_token.type != TokenType.COLON
               and self.current_token.type != TokenType.NEWLINE):
            if self.current_token.type == TokenType.SEMICOLON:
                self.eat(TokenType.SEMICOLON)
                node.end = ''
            else:
                node.nodes.append(self.parse_expr())
                node.end = '\n'

        return node

    def parse_input_statement(self):
        """
        input_statement : INPUT (STRING SEMICOLON)* ID (COMMA ID)*
        """
        self.eat(TokenType.INPUT)
        node = InputNode()

        if self.current_token.type == TokenType.STRING:
            node.prompt = self.current_token.value
            self.eat(TokenType.STRING)
            self.eat(TokenType.SEMICOLON)

        node.ids.append(self.parse_variable())

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            node.ids.append(self.parse_variable())

        return node

    def parse_get_statement(self):
        """
        get_statement : GET ID
        """
        self.eat(TokenType.GET)
        var = VarNode(self.current_token)
        self.eat(TokenType.ID)
        return GetNode(var)

    def parse_on_statement(self):
        """
        on_statement   : ON expr GOTO expr (expr)*
                       | ON expr GOSUB expr (expr)*
        """
        self.eat(TokenType.ON)
        expr = self.parse_expr()
        node = OnNode(expr, self.current_token)

        if self.current_token.type == TokenType.GOTO:
            self.eat(TokenType.GOTO)
        else:
            self.eat(TokenType.GOSUB)

        node.linenos.append(self.parse_expr())

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            node.linenos.append(self.parse_expr())

        return node

    def parse_goto_statement(self):
        """
        goto_statement : GOTO LINENUM
        """
        self.eat(TokenType.GOTO)
        return GoToNode(self.parse_expr())

    def parse_gosub_statement(self):
        """
        gosub_statement : GOSUB LINENUM
        """
        self.eat(TokenType.GOSUB)
        return GoSubNode(self.parse_expr())

    def parse_return_statement(self):
        """
        return_statement : RETURN
        """
        node = ReturnNode(self.current_token)
        self.eat(TokenType.RETURN)
        return node

    def parse_for_statement(self):
        """
        for_statement : FOR assignment_statement TO limit (STEP step)
        """
        self.eat(TokenType.FOR)
        var = VarNode(self.current_token)
        assign = self.parse_assignment_statement()
        self.eat(TokenType.TO)
        limit = self.parse_expr()
        step = None
        if self.current_token.type == TokenType.STEP:
            self.eat(TokenType.STEP)
            step = self.parse_expr()

        return ForNode(assign, var, limit, step)

    def parse_next_statement(self):
        """
        next_statement : NEXT
                       | NEXT ID (COMMA ID)*
        """
        nodes = []

        node = NextNode(self.current_token)
        self.eat(TokenType.NEXT)

        if self.current_token.type == TokenType.ID:
            self.eat(TokenType.ID)

            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                nodes.append(NextNode(self.current_token))
                self.eat(TokenType.ID)

        return node, nodes

    def parse_dim_statement(self):
        """
        dim_statement : DIM ID LPAREN expr (COMMA expr)* RPAREN
        """
        self.eat(TokenType.DIM)
        var = VarNode(self.current_token)
        self.eat(TokenType.ID)
        node = DimNode(var)

        self.eat(TokenType.LPAREN)

        node.dims.append(self.parse_expr())

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            node.dims.append(self.parse_expr())

        self.eat(TokenType.RPAREN)

        return node

    def parse_if_statement(self):
        """
        if_statement   : IF expr THEN then_statement
                       | IF expr THEN then_statement ELSE else_statement
        then_statement : LINENUM
                       | statement_list
        else_statement : LINENUM
                       | statement_list
        """
        self.eat(TokenType.IF)
        expr = self.parse_expr()
        self.eat(TokenType.THEN)
        if self.current_token.type == TokenType.NUMBER:
            nodes = [GoToNode(self.parse_expr())]
        else:
            nodes = self.parse_statement_list()
        skip_then = len(nodes)

        if self.current_token.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            if self.current_token.type == TokenType.NUMBER:
                else_nodes = [GoToNode(self.parse_expr())]
            else:
                else_nodes = self.parse_statement_list()
            skip_else = len(else_nodes)
            nodes.append(SkipNode(skip_else))
            skip_then += 1
            nodes += else_nodes

        return IfNode(expr, skip_then), nodes

    def parse_data_statement(self):
        """
        data_statement : DATA constant (COMMA constant)*
        constant       : NUMBER
                       | STRING
        """
        self.eat(TokenType.DATA)

        if (self.current_token.type == TokenType.NUMBER
                or self.current_token.type == TokenType.STRING):
            self.global_data.append(self.current_token.value)
            self.eat(self.current_token.type)
        else:
            unexpected_token_error(self.current_token)

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            if (self.current_token.type == TokenType.NUMBER
                    or self.current_token.type == TokenType.STRING):
                self.global_data.append(self.current_token.value)
                self.eat(self.current_token.type)
            else:
                unexpected_token_error(self.current_token)

        return NoOpNode()

    def parse_variable(self):
        """
        variable : ID
                 | ID LPAREN expr (COMMA expr)* RPAREN
        """
        node = VarNode(self.current_token)
        self.eat(TokenType.ID)

        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)

            node.idxs.append(self.parse_expr())

            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                node.idxs.append(self.parse_expr())

            self.eat(TokenType.RPAREN)

        return node

    def parse_read_statement(self):
        """
        read_statement : READ ID (COMMA ID)*
        """
        self.eat(TokenType.READ)
        node = ReadNode()

        node.ids.append(self.parse_variable())

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            node.ids.append(self.parse_variable())

        return node

    def parse_restore_statement(self):
        """
        restore_statement : RESTORE
        """
        self.eat(TokenType.RESTORE)
        return RestoreNode()

    def parse_end_statement(self):
        """
        end_statement : END
        """
        self.eat(TokenType.END)
        return EndNode()

    def parse_function(self):
        """
        function : ABS LPAREN expr RPAREN
                 | ASC LPAREN expr RPAREN
                 | ATN LPAREN expr RPAREN
                 | CHR$ LPAREN expr RPAREN
                 | COS LPAREN expr RPAREN
                 | EXP LPAREN expr RPAREN
                 | INT LPAREN expr RPAREN
                 | LEN LPAREN expr RPAREN
                 | LOG LPAREN expr RPAREN
                 | RND LPAREN expr RPAREN
                 | SGN LPAREN expr RPAREN
                 | SIN LPAREN expr RPAREN
                 | SPC LPAREN expr RPAREN
                 | SQR LPAREN expr RPAREN
                 | STR$ LPAREN expr RPAREN
                 | TAN LPAREN expr RPAREN
                 | TAB LPAREN expr RPAREN
                 | VAL LPAREN expr RPAREN
        """
        token = self.current_token
        self.eat(token.type)
        self.eat(TokenType.LPAREN)
        node = UnaryOpNode(token, self.parse_expr())
        self.eat(TokenType.RPAREN)

        return node

    def parse_empty(self):
        return NoOpNode()

    def parse_value(self):
        """
        value : NUMBER
              | STRING
              | LPAREN expr RPAREN
              | function LPAREN expr RPAREN
              | variable
        """
        token = self.current_token
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            node = NumNode(token)
        elif token.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            node = StringNode(token)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.parse_expr()
            self.eat(TokenType.RPAREN)
        elif token.type == TokenType.LEFT:
            self.eat(TokenType.LEFT)
            self.eat(TokenType.LPAREN)
            string = self.parse_expr()
            self.eat(TokenType.COMMA)
            length = self.parse_expr()
            self.eat(TokenType.RPAREN)
            node = LeftDollarNode(string, length)
        elif token.type == TokenType.MID:
            self.eat(TokenType.MID)
            self.eat(TokenType.LPAREN)
            string = self.parse_expr()
            self.eat(TokenType.COMMA)
            start = self.parse_expr()
            self.eat(TokenType.COMMA)
            length = self.parse_expr()
            self.eat(TokenType.RPAREN)
            node = MidDollarNode(string, start, length)
        elif token.type == TokenType.RIGHT:
            self.eat(TokenType.RIGHT)
            self.eat(TokenType.LPAREN)
            string = self.parse_expr()
            self.eat(TokenType.COMMA)
            length = self.parse_expr()
            self.eat(TokenType.RPAREN)
            node = RightDollarNode(string, length)
        elif token.type in (TokenType.ABS, TokenType.ASC, TokenType.ATN,
                            TokenType.CHR, TokenType.COS, TokenType.EXP,
                            TokenType.INT, TokenType.LEN, TokenType.LOG,
                            TokenType.RND, TokenType.SGN, TokenType.SIN,
                            TokenType.SPC, TokenType.SQR, TokenType.STR,
                            TokenType.TAB, TokenType.TAN, TokenType.VAL):
            node = self.parse_function()
        else:
            node = self.parse_variable()

        return node

    def parse_power_expr(self):
        """
        power_expr : value (POWER value)*
        """
        node = self.parse_value()

        while self.current_token.type == TokenType.POWER:
            token = self.current_token
            self.eat(TokenType.POWER)

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
            self.eat(TokenType.PLUS)
            node = UnaryOpNode(token, self.parse_power_expr())
        elif token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            node = UnaryOpNode(token, self.parse_power_expr())
        else:
            node = self.parse_power_expr()

        return node

    def parse_mult_expr(self):
        """
        mult_expr : unary_expr ((MUL | DIV) unary_expr)*
        """
        node = self.parse_unary_expr()

        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            token = self.current_token
            self.eat(token.type)

            node = BinOpNode(left=node,
                             op=token,
                             right=self.parse_unary_expr())

        return node

    def parse_add_expr(self):
        """
        add_expr : mult_expr ((PLUS | MINUS) mult_expr)*
        """
        node = self.parse_mult_expr()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            self.eat(token.type)

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
            self.eat(token.type)

            node = BinOpNode(left=node, op=token, right=self.parse_add_expr())

        return node

    def parse_not_expr(self):
        """
        not_expr : (NOT)* compare_expr
        """
        token = self.current_token
        if token.type == TokenType.NOT:
            self.eat(TokenType.NOT)
            node = UnaryOpNode(token, self.parse_compare_expr())
        else:
            node = self.parse_compare_expr()

        return node

    def parse_and_expr(self):
        """
        and_expr : not_expr (AND not_expr)*
        """
        node = self.parse_not_expr()

        while self.current_token.type == TokenType.AND:
            token = self.current_token
            self.eat(token.type)

            node = BinOpNode(left=node, op=token, right=self.parse_not_expr())

        return node

    def parse_expr(self):
        """
        Expression Parser

        expr         : and_expr (OR and_expr)*
        and_expr     : not_expr (AND not_expr)*
        not_expr     : (NOT)* compare_expr
        compare_expr : add_expr ((EQUAL | NOTEQUAL | GREATER | LESS |
                                  GEQUAL | LEQUAL) add_expr)*
        add_expr     : mult_expr ((PLUS | MINUS) mult_expr)*
        mult_expr    : unary_expr ((MUL | DIV) unary_expr)*
        unary_expr   : (PLUS | MINUS)* power_expr
        power_expr   : value (POWER value)*
        value        : NUMBER | STRING | LPAREN expr RPAREN |
                       variable | function LPAREN expr RPAREN
        """
        node = self.parse_and_expr()

        while self.current_token.type == TokenType.OR:
            token = self.current_token
            self.eat(token.type)

            node = BinOpNode(left=node, op=token, right=self.parse_and_expr())

        return node

    def parse(self):
        node = self.parse_program()
        if self.current_token.type != TokenType.EOF:
            raise ParserError("EOF not found")

        return node, self.global_data


###############################################################################
#                                                                             #
#  AST NODE VISITOR                                                           #
#                                                                             #
###############################################################################


class NodeVisitor():
    """ AST Node Visitor """

    def visit(self, node):
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################


class Interpreter(NodeVisitor):
    """ BASIC Interpreter """

    def __init__(self, tree, global_data):
        self.tree = tree
        self.next_node = 0
        self.return_stack = []
        self.for_stack = []
        self.line_numbers = {}
        self.global_scope = {}
        self.global_data = global_data
        self.global_data_ptr = 0

    def visit_ProgramNode(self, node):
        self.line_numbers = node.line_numbers

        current_node = 0
        while True:
            self.next_node = current_node + 1
            self.visit(node.nodes[current_node])
            current_node = self.next_node
            if current_node is None or current_node >= len(node.nodes):
                break

    def visit_ClsNode(self, node):
        os.system("clear")
        os.system("clear")

    def visit_PrintNode(self, node):
        s = ""
        for n in node.nodes:
            val = self.visit(n)
            if type(val) is str:
                s += f"{val}"
            else:
                s += f"{val:g}"
        print(s, end=node.end)

    def visit_InputNode(self, node):
        values = input(f"{node.prompt}? ").split(',')
        values = [v.strip() for v in values]
        values = [v.upper() for v in values]

        for i in range(len(node.ids)):
            id = node.ids[i]
            value = values[i] if i < len(values) else None

            var_name = id.value

            if value is None or value == "":
                value = "" if var_name.endswith('$') else "0"

            if not var_name.endswith('$'):
                try:
                    value = float(value)
                except ValueError:
                    type_mismatch_error(id.token)

            var = self.global_scope.get(var_name)

            # Attempt to assign a non-dim'd array
            if var is None and len(id.idxs) > 0:
                dims = [11 for _ in range(len(id.idxs))]
                val = "" if var_name.endswith('$') else 0.0

                self.global_scope[var_name] = init_list(dims, val)
                var = self.global_scope.get(var_name)

            if var is not None and type(var) is list:
                idxs = []
                tokens = []
                for idx in id.idxs:
                    idxs.append(int(self.visit(idx)))
                    tokens.append(idx.token)

                set_list(self.global_scope[var_name], value, idxs, tokens)
            else:
                self.global_scope[var_name] = value

    def visit_GetNode(self, node):
        var_name = node.var.value
        try:
            value = getkey()
        except (KeyboardInterrupt, SystemExit):
            os.system("stty sane")

        if not var_name.endswith('$'):
            type_mismatch_error(node.var.token)

        var = self.global_scope.get(var_name)

        # Attempt to assign a non-dim'd array
        if var is None and len(node.var.idxs) > 0:
            dims = [11 for _ in range(len(node.var.idxs))]
            val = "" if var_name.endswith('$') else 0.0

            self.global_scope[var_name] = init_list(dims, val)
            var = self.global_scope.get(var_name)

        if var is not None and type(var) is list:
            idxs = []
            tokens = []
            for idx in node.var.idxs:
                idxs.append(int(self.visit(idx)))
                tokens.append(idx.token)

            set_list(self.global_scope[var_name], value, idxs, tokens)
        else:
            self.global_scope[var_name] = value

    def visit_AssignNode(self, node):
        var_name = node.left.value
        value = self.visit(node.right)

        if var_name.endswith('$') and type(value) is not str:
            type_mismatch_error(node.right.token)

        if not var_name.endswith('$') and type(value) is str:
            type_mismatch_error(node.right.token)

        var = self.global_scope.get(var_name)

        # Attempt to assign a non-dim'd array
        if var is None and len(node.left.idxs) > 0:
            dims = [11 for _ in range(len(node.left.idxs))]
            val = "" if var_name.endswith('$') else 0.0

            self.global_scope[var_name] = init_list(dims, val)
            var = self.global_scope.get(var_name)

        if var is not None and type(var) is list:
            idxs = []
            tokens = []
            for idx in node.left.idxs:
                idxs.append(int(self.visit(idx)))
                tokens.append(idx.token)

            set_list(self.global_scope[var_name], value, idxs, tokens)
        else:
            self.global_scope[var_name] = value

    def visit_VarNode(self, node):
        var_name = node.value
        var = self.global_scope.get(var_name)

        if var is None:
            identifier_not_found_error(node.token)

        if type(var) is list:
            idxs = []
            tokens = []
            for idx in node.idxs:
                idxs.append(int(self.visit(idx)))
                tokens.append(idx.token)

            value = get_list(var, idxs, tokens)
        else:
            value = var

        return value

    def visit_NoOpNode(self, node):
        pass

    def visit_BinOpNode(self, node):
        left_value = self.visit(node.left)
        right_value = self.visit(node.right)

        if node.op.type == TokenType.PLUS:
            if type(left_value) is not type(right_value):
                type_mismatch_error(node.right.token)
            else:
                return left_value + right_value

        elif node.op.type == TokenType.MINUS:
            if type(left_value) is str:
                type_mismatch_error(node.left.token)
            elif type(right_value) is str:
                type_mismatch_error(node.right.token)
            else:
                return left_value - right_value

        elif node.op.type == TokenType.MUL:
            if type(left_value) is str:
                type_mismatch_error(node.left.token)
            elif type(right_value) is str:
                type_mismatch_error(node.right.token)
            else:
                return left_value * right_value

        elif node.op.type == TokenType.DIV:
            if type(left_value) is str:
                type_mismatch_error(node.left.token)
            elif type(right_value) is str:
                type_mismatch_error(node.right.token)
            else:
                return left_value / right_value

        elif node.op.type == TokenType.POWER:
            if type(left_value) is str:
                type_mismatch_error(node.left.token)
            elif type(right_value) is str:
                type_mismatch_error(node.right.token)
            else:
                return left_value**right_value

        elif node.op.type == TokenType.EQUAL:
            if type(left_value) is not type(right_value):
                type_mismatch_error(node.right.token)
            else:
                return 1.0 if left_value == right_value else 0.0

        elif node.op.type == TokenType.NOTEQUAL:
            if type(left_value) is not type(right_value):
                type_mismatch_error(node.right.token)
            else:
                return 1.0 if left_value != right_value else 0.0

        elif node.op.type == TokenType.GREATER:
            if type(left_value) is not type(right_value):
                type_mismatch_error(node.right.token)
            else:
                return 1.0 if left_value > right_value else 0.0

        elif node.op.type == TokenType.LESS:
            if type(left_value) is not type(right_value):
                type_mismatch_error(node.right.token)
            else:
                return 1.0 if left_value < right_value else 0.0

        elif node.op.type == TokenType.GEQUAL:
            if type(left_value) is not type(right_value):
                type_mismatch_error(node.right.token)
            else:
                return 1.0 if left_value >= right_value else 0.0

        elif node.op.type == TokenType.LEQUAL:
            if type(left_value) is not type(right_value):
                type_mismatch_error(node.right.token)
            else:
                return 1.0 if left_value <= right_value else 0.0

        elif node.op.type == TokenType.AND:
            if type(left_value) is str:
                type_mismatch_error(node.left.token)
            elif type(right_value) is str:
                type_mismatch_error(node.right.token)
            else:
                return (1.0 if
                        (left_value != 0.0) and (right_value != 0.0) else 0.0)

        elif node.op.type == TokenType.OR:
            if type(left_value) is str:
                type_mismatch_error(node.left.token)
            elif type(right_value) is str:
                type_mismatch_error(node.right.token)
            else:
                return (1.0 if
                        (left_value != 0.0) or (right_value != 0.0) else 0.0)

    def visit_LeftDollarNode(self, node):
        string = self.visit(node.string)
        if type(string) is not str:
            type_mismatch_error(node.string.token)

        length = self.visit(node.length)
        if type(length) is str:
            type_mismatch_error(node.length.token)

        length = int(length)

        if length < 1 or length > len(string):
            out_of_range_error(node.length.token)

        return string[:length]

    def visit_MidDollarNode(self, node):
        string = self.visit(node.string)
        if type(string) is not str:
            type_mismatch_error(node.string.token)

        start = self.visit(node.start)
        if type(start) is str:
            type_mismatch_error(node.start.token)

        start = int(start)

        if start < 1 or start > len(string):
            out_of_range_error(node.start.token)

        length = self.visit(node.length)
        if type(length) is str:
            type_mismatch_error(node.length.token)

        length = int(length)

        if length < 1 or start + length - 1 > len(string):
            out_of_range_error(node.length.token)

        return string[start - 1:start - 1 + length]

    def visit_RightDollarNode(self, node):
        string = self.visit(node.string)
        if type(string) is not str:
            type_mismatch_error(node.string.token)

        length = self.visit(node.length)
        if type(length) is str:
            type_mismatch_error(node.length.token)

        length = int(length)

        if length < 1 or length > len(string):
            out_of_range_error(node.length.token)

        return string[-length:]

    def visit_UnaryOpNode(self, node):
        value = self.visit(node.expr)

        op = node.op.type
        if op == TokenType.PLUS:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return +value

        elif op == TokenType.MINUS:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return -value

        elif op == TokenType.NOT:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return 1.0 if not (value != 0.0) else 0.0

        elif op == TokenType.ABS:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return abs(value)

        elif op == TokenType.ASC:
            if type(value) is not str:
                type_mismatch_error(node.expr.token)
            elif len(value) < 1:
                empty_string_error(node.expr.token)
            else:
                return float(ord(value[0]))

        elif op == TokenType.ATN:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return math.atan(value)

        elif op == TokenType.CHR:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return chr(int(value))

        elif op == TokenType.COS:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return math.cos(value)

        elif op == TokenType.EXP:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return math.exp(value)

        elif op == TokenType.INT:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return float(math.floor(value))

        elif op == TokenType.LEN:
            if type(value) is not str:
                type_mismatch_error(node.expr.token)
            else:
                return len(value)

        elif op == TokenType.LOG:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return math.log(value)

        elif op == TokenType.RND:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                if value < 0:
                    random.seed(abs(value))
                return random.random()

        elif op == TokenType.SGN:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return 0.0 if value == 0.0 else math.copysign(1.0, value)

        elif op == TokenType.SIN:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return math.sin(value)

        elif op == TokenType.SPC:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return f"\033[{int(value)}C"

        elif op == TokenType.SQR:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return math.sqrt(value)

        elif op == TokenType.STR:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return str(value)

        elif op == TokenType.TAB:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return f"\033[{int(value)}G"

        elif op == TokenType.TAN:
            if type(value) is str:
                type_mismatch_error(node.expr.token)
            else:
                return math.tan(value)

        elif op == TokenType.VAL:
            if type(value) is not str:
                type_mismatch_error(node.expr.token)
            elif len(value) < 1:
                empty_string_error(node.expr.token)
            else:
                s = ""
                p = 0
                while (value[p] is not None and not value[p].isdigit()
                       and value[p] != '.' and value[p] != '-'):
                    p += 1
                while (value[p] is not None
                       and (value[p].isdigit() or value[p] == '.'
                            or value[p] == '-')):
                    s += value[p]
                    p += 1
                return float(s)

    def visit_NumNode(self, node):
        return node.value

    def visit_StringNode(self, node):
        return node.value

    def visit_OnNode(self, node):
        test = self.visit(node.expr)
        if type(test) is str:
            type_mismatch_error(node.expr.token)

        test = int(test) - 1

        if test < 0 or test >= len(node.linenos):
            return

        line_number = self.visit(node.linenos[test])
        if type(line_number) is str:
            type_mismatch_error(node.linenos[test].token)

        if node.token.type == TokenType.GOSUB:
            self.return_stack.append(self.next_node)

        self.next_node = self.line_numbers.get(int(line_number))
        if self.next_node is None:
            line_not_found_error(node.linenos[test].token)

    def visit_GoToNode(self, node):
        line_number = self.visit(node.expr)
        if type(line_number) is str:
            type_mismatch_error(node.expr.token)

        self.next_node = self.line_numbers.get(int(line_number))
        if self.next_node is None:
            line_not_found_error(node.expr.token)

    def visit_GoSubNode(self, node):
        line_number = self.visit(node.expr)
        if type(line_number) is str:
            type_mismatch_error(node.expr.token)

        self.return_stack.append(self.next_node)
        self.next_node = self.line_numbers.get(int(line_number))
        if self.next_node is None:
            line_not_found_error(node.expr.token)

    def visit_ReturnNode(self, node):
        try:
            self.next_node = self.return_stack.pop()
        except IndexError:
            return_without_gosub_error(node.token)

    def visit_ForNode(self, node):
        self.visit(node.assign)

        self.for_stack.append(
            (node.var, node.limit, node.step, self.next_node))

    def visit_NextNode(self, node):
        try:
            for_node = self.for_stack.pop()
        except IndexError:
            next_without_for_error(node.token)
        else:
            var_name = for_node[0].value
            limit = self.visit(for_node[1])
            if for_node[2]:
                step = self.visit(for_node[2])
            else:
                step = 1.0

            self.global_scope[var_name] += step

            value = self.global_scope[var_name]

            if ((step > 0.0 and value <= limit)
                    or (step < 0.0 and value >= limit)):
                self.for_stack.append(for_node)
                self.next_node = for_node[3]

    def visit_IfNode(self, node):
        test = self.visit(node.expr)
        if type(test) is str:
            type_mismatch_error(node.expr.token)

        if not test:
            self.next_node += node.skip_then

    def visit_SkipNode(self, node):
        self.next_node += node.skip

    def visit_ReadNode(self, node):
        for id in node.ids:
            if self.global_data_ptr >= len(self.global_data):
                out_of_data_error(id.token)

            var_name = id.value
            value = self.global_data[self.global_data_ptr]

            if var_name.endswith('$') and type(value) is not str:
                type_mismatch_error(id.token)

            if not var_name.endswith('$') and type(value) is str:
                type_mismatch_error(id.token)

            var = self.global_scope.get(var_name)

            # Attempt to assign a non-dim'd array
            if var is None and len(id.idxs) > 0:
                dims = [11 for _ in range(len(id.idxs))]
                val = "" if var_name.endswith('$') else 0.0

                self.global_scope[var_name] = init_list(dims, val)
                var = self.global_scope.get(var_name)

            if var is not None and type(var) is list:
                idxs = []
                tokens = []
                for idx in id.idxs:
                    idxs.append(int(self.visit(idx)))
                    tokens.append(idx.token)

                set_list(self.global_scope[var_name], value, idxs, tokens)
            else:
                self.global_scope[var_name] = value

            self.global_data_ptr += 1

    def visit_RestoreNode(self, node):
        self.global_data_ptr = 0

    def visit_DimNode(self, node):
        var_name = node.var.value
        if self.global_scope.get(var_name):
            redimd_array_error(node.var)

        dims = []
        for dim in node.dims:
            dims.append(int(self.visit(dim)) + 1)

        val = "" if var_name.endswith('$') else 0.0

        self.global_scope[var_name] = init_list(dims, val)

    def visit_EndNode(self, node):
        self.next_node = None

    def run(self):
        tree = self.tree
        if tree is None:
            return ""
        return self.visit(tree)


def main():
    arg_parser = argparse.ArgumentParser(description="BASIC Interpreter")
    arg_parser.add_argument("filename", help="BASIC source file")

    args = arg_parser.parse_args()

    with open(args.filename, 'r') as file:
        text = file.read()

        lexer = Lexer(text)
        try:
            parser = Parser(lexer)
            tree, global_data = parser.parse()
        except (LexerError, ParserError) as e:
            print(e.message)
            sys.exit(1)

        interpreter = Interpreter(tree, global_data)
        try:
            interpreter.run()
        except RuntimeError as e:
            print(e.message)
            sys.exit(1)


if __name__ == "__main__":
    main()
