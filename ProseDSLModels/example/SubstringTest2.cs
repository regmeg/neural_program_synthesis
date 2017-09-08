/*
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ProgramSynthesis;
using Microsoft.ProgramSynthesis.Compiler;
using Microsoft.ProgramSynthesis.Learning;
using Microsoft.ProgramSynthesis.Specifications;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Microsoft.ProgramSynthesis;
using Microsoft.ProgramSynthesis.AST;
using Microsoft.ProgramSynthesis.Compiler;
using System.Collections.Generic;
using Microsoft.ProgramSynthesis.Specifications;
using Microsoft.ProgramSynthesis.Learning;
using System.Linq;

namespace ProseTutorial
{
    [TestClass]
    public class SubstringTest
    {
        string grammar_path = "../../../grammar/substring.grammar";
        [TestMethod]
        public void TestSubstringProgram()
        {
            var grammar = DSLCompiler.
                ParseGrammarFromFile(grammar_path);
            //var program = grammar.Value.ParseAST("Substring(x, AbsPos(7), AbsPos(15))", 
            //    ASTSerializationFormat.HumanReadable);
            var program = ProgramNode.Parse("Substring(x, AbsPos(7), AbsPos(15))", grammar.Value, ASTSerializationFormat.HumanReadable);

            var input = State.Create(grammar.Value.InputSymbol, "Bjoern Hartmann");
            var output = program.Invoke(input) as string; 
            Assert.AreEqual("Hartmann", output);
        }

        [TestMethod] public void TestLearnSubstringSingleExample()
        {
            var grammar = DSLCompiler.
                ParseGrammarFromFile(grammar_path);

            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            var input = State.Create(grammar.Value.InputSymbol, "Bjoern Hartmann");
            var examples = new Dictionary<State, object> { { input, "Hartmann" }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = learnedSet.RealizedPrograms.First().Invoke(input) as string;
            Assert.AreEqual("Hartmann", output);
        }

        [TestMethod]
        public void TestLearnSubstring()
        {
            var grammar = DSLCompiler.
                ParseGrammarFromFile(grammar_path);

            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            var input = State.Create(grammar.Value.InputSymbol, "Bjoern Hartmann");
            var input2 = State.Create(grammar.Value.InputSymbol, "Andrew Head");
            var examples = new Dictionary<State, object> { { input, "Hartmann" }, {input2, "Head"} };
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = learnedSet.RealizedPrograms.First().Invoke(input) as string;
            Assert.AreEqual("Hartmann", output);
            var output2 = learnedSet.RealizedPrograms.First().Invoke(input2) as string;
            Assert.AreEqual("Head", output2);
        }


        [TestMethod]
        public void TestLearnStringTransformation()
        {
            var grammar = DSLCompiler.
                ParseGrammarFromFile(grammar_path);

            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            var input = State.Create(grammar.Value.InputSymbol, "Bjoern Hartmann");
            var examples = new Dictionary<State, object> { { input, "Hartmann" }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = learnedSet.RealizedPrograms.First().Invoke(input) as string;
            Assert.AreEqual("Hartmann, B.", output);
        }
    }
}
*/