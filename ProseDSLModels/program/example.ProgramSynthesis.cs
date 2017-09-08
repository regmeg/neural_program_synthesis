/*
using Microsoft.ProgramSynthesis;
using Microsoft.ProgramSynthesis.AST;
using Microsoft.ProgramSynthesis.Compiler;

// Parse the DSL grammar above, saved in a .grammar file
var grammar = DSLCompiler.ParseGrammarFromFile("SubstringExtraction.grammar").Value;
// Parse a program in this grammar. PROSE supports two serialization formats:
// "human-readable" expression format, used in this tutorial, and machine-readable XML.
var ast =
  ProgramNode.Parse("Substring(x, PositionPair(AbsolutePosition(x, 0), AbsolutePosition(x, 5)))",
                    grammar, ASTSerializationFormat.HumanReadable);
// Create an input state to the program. It contains one binding: a variable 'x' (DSL input)
// is bound to the string "PROSE Rocks".
var input = State.Create(grammar.InputSymbol, "PROSE Rocks");
// Execute the program on the input state.
var output = (string) ast.Invoke(input);
Assert(output == "PROSE");
*/