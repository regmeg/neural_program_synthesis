using System;
using Microsoft.ProgramSynthesis;
using Microsoft.ProgramSynthesis.AST;
using Microsoft.ProgramSynthesis.Diagnostics;
using Microsoft.ProgramSynthesis.Compiler;
using System.Collections.Generic;
using Microsoft.ProgramSynthesis.Specifications;
using Microsoft.ProgramSynthesis.Learning;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Diagnostics; 
namespace ProseDSLModels
{
    [TestClass]
    public class Tests
    {
        private TestContext testContextInstance;

        /// <summary>
        ///  Gets or sets the test context which provides
        ///  information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get { return testContextInstance; }
            set { testContextInstance = value; }
        }

        [TestMethod]
        public void testAdd(){
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);

            var ast = grammar.Value.ParseAST("add(stall(stall(stall(stall(x)))))", ASTSerializationFormat.HumanReadable);

      		double[] numbers = new double[4] {10.9, 15.0, -14.5, 12.3};

    		var input = State.Create(grammar.Value.InputSymbol, numbers);

    		var output = (double[]) ast.Invoke(input);
    		Assert.AreEqual(23.7, output[0], 0.001d);
        }

        [TestMethod]
        public void TestLearnAdd()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = add(inp1);
            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            stdoutprogram(learnedSet.RealizedPrograms.First().ToString(), "Add");
            Assert.AreEqual(23.7, output[0], 0.001d);
            TestContext.WriteLine("Running random exmaples for LearnAdd");
            //test 1000 random examples
            for (int i=0;i<1000;++i) {
                var iTmp = genRnd();
                var inpTmp = State.Create(grammar.Value.InputSymbol, iTmp);
                var oTmp = add(iTmp);
                var outTmp = (double[]) learnedSet.RealizedPrograms.First().Invoke(inpTmp);
                TestContext.WriteLine("Excpt [{0}]", string.Join(", ", oTmp));
                TestContext.WriteLine("Actul [{0}]", string.Join(", ", outTmp));
                Assert.AreEqual(oTmp[0], outTmp[0], 0.001d);
            }

        }

        [TestMethod]
        public void TestLearnStall()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);
            //{30.3, -8.9, -19.0, -2.4};
            /*
            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = stall(inp1);
            var input1 = State.Create(grammar.Value.InputSymbol, inp1);
            double[] inp2 = new double[4] {30.3, -8.9, -19.0, -2.4};
            double[] out2 = stall(inp1);
            var input2 = State.Create(grammar.Value.InputSymbol, inp1);
            var examples = new Dictionary<State, object> { { input1, out1 },  { input2, out2 }};
            */
            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = stall(inp1);

            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            stdoutprogram(learnedSet.RealizedPrograms.First().ToString(), "Stall");

            TestContext.WriteLine("Running random exmaples for Stall");
            //test 1000 random examples
            for (int i=0;i<1000;++i) {
                var iTmp = genRnd();
                var inpTmp = State.Create(grammar.Value.InputSymbol, iTmp);
                var oTmp = stall(iTmp);
                var outTmp = (double[]) learnedSet.RealizedPrograms.First().Invoke(inpTmp);
                TestContext.WriteLine("Excpt [{0}]", string.Join(", ", oTmp));
                TestContext.WriteLine("Actul [{0}]", string.Join(", ", outTmp));
                Assert.AreEqual(oTmp[0], outTmp[0], 0.001d);
                Assert.AreEqual(oTmp[1], outTmp[1], 0.001d);
                Assert.AreEqual(oTmp[2], outTmp[2], 0.001d);
                Assert.AreEqual(oTmp[3], outTmp[3], 0.001d);
            }

        }

        [TestMethod]
        public void testMult(){
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);

            var ast = grammar.Value.ParseAST("mult(stall(stall(stall(stall(x)))))", ASTSerializationFormat.HumanReadable);

            double[] numbers = new double[4] {10.9, 15.0, -14.5, 12.3};

            var input = State.Create(grammar.Value.InputSymbol, numbers);

            var output = (double[]) ast.Invoke(input);
            Assert.AreEqual(-29160.225, output[0], 0.001d);
        }


        [TestMethod]
        public void TestLearnMult()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = mult(inp1);
            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            stdoutprogram(learnedSet.RealizedPrograms.First().ToString(), "Mult");
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            Assert.AreEqual(-29160.225, output[0], 0.001d);
            //test 1000 random examples
            for (int i=0;i<1000;++i) {
                var iTmp = genRnd();
                var inpTmp = State.Create(grammar.Value.InputSymbol, iTmp);
                var oTmp = mult(iTmp);
                var outTmp = (double[]) learnedSet.RealizedPrograms.First().Invoke(inpTmp);
                TestContext.WriteLine("Excpt [{0}]", string.Join(", ", oTmp));
                TestContext.WriteLine("Actul [{0}]", string.Join(", ", outTmp));
                Assert.AreEqual(oTmp[0], outTmp[0], 0.001d);
            }
        }

        [TestMethod]
        public void testLen(){
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);

            var ast = grammar.Value.ParseAST("len(stall(stall(stall(stall(x)))))", ASTSerializationFormat.HumanReadable);

            double[] numbers = new double[4] {10.9, 15.0, -14.5, 12.3};

            var input = State.Create(grammar.Value.InputSymbol, numbers);

            var output = (double[]) ast.Invoke(input);
            Assert.AreEqual(4, output[0], 0.001d);
        }

        [TestMethod]
        public void TestLearnLen()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = new double[4] {4, 0.0, 0.0, 0.0};
            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            Assert.AreEqual(4, output[0], 0.001d);
        }

        [TestMethod]
        public void testDiv(){
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);

            var ast = grammar.Value.ParseAST("stall(stall(stall(stall(div(x, len(x))))))", ASTSerializationFormat.HumanReadable);

            double[] numbers = new double[4] {10.9, 15.0, -14.5, 12.3};

            var input = State.Create(grammar.Value.InputSymbol, numbers);

            var output = (double[]) ast.Invoke(input);
            Assert.AreEqual(2.725, output[0], 0.001d);
        }

        [TestMethod]
        public void TestLearnDiv()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = new double[4] {2.725, 0.0, 0.0, 0.0};
            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            Assert.AreEqual(2.725, output[0], 0.001d);
        }


        [TestMethod]
        public void testSub(){
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);

            var ast = grammar.Value.ParseAST("stall(stall(stall(stall(sub(x, stall(x))))))", ASTSerializationFormat.HumanReadable);

            double[] numbers = new double[4] {10.9, 15.0, -14.5, 12.3};

            var input = State.Create(grammar.Value.InputSymbol, numbers);

            var output = (double[]) ast.Invoke(input);
            Assert.AreEqual(0.0,   output[0], 0.001d);
            Assert.AreEqual(4.1 ,  output[1], 0.001d);
            Assert.AreEqual(-25.4, output[2], 0.001d);
            Assert.AreEqual(1.4,   output[3], 0.001d);
        }

        [TestMethod]
        public void TestLearnSub()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = new double[4] {0.0, 4.1, -25.4, 1.4};
            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            Assert.AreEqual(0.0,   output[0], 0.001d);
            Assert.AreEqual(4.1 ,  output[1], 0.001d);
            Assert.AreEqual(-25.4, output[2], 0.001d);
            Assert.AreEqual(1.4,   output[3], 0.001d);
        }

        [TestMethod]
        public void testAvgVal(){
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);

            var ast = grammar.Value.ParseAST("stall(stall(stall(div(add(x), len(x))))))", ASTSerializationFormat.HumanReadable);

            double[] numbers = new double[4] {10.9, 15.0, -14.5, 12.3};

            var input = State.Create(grammar.Value.InputSymbol, numbers);

            var output = (double[]) ast.Invoke(input);
            Assert.AreEqual(5.925, output[0], 0.001d);
        }

        [TestMethod]
        public void TestLearnAvgVal()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = avg_val(inp1);
            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            stdoutprogram(learnedSet.RealizedPrograms.First().ToString(), "AvgVal");
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            Assert.AreEqual(5.925, output[0], 0.001d);
            //test 1000 random examples
            for (int i=0;i<1000;++i) {
                var iTmp = genRnd();
                var inpTmp = State.Create(grammar.Value.InputSymbol, iTmp);
                var oTmp = avg_val(iTmp);
                var outTmp = (double[]) learnedSet.RealizedPrograms.First().Invoke(inpTmp);
                TestContext.WriteLine("Excpt [{0}]", string.Join(", ", oTmp));
                TestContext.WriteLine("Actul [{0}]", string.Join(", ", outTmp));
                Assert.AreEqual(oTmp[0], outTmp[0], 0.001d);
            }
        }

        [TestMethod]
        public void testCenter(){
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);

            var ast = grammar.Value.ParseAST("stall(stall(sub(div(add(x), len(x)), stall(x)))))", ASTSerializationFormat.HumanReadable);
            double[] numbers = new double[4] {10.9, 15.0, -14.5, 12.3};
            //4.975, 9.075, -20.425, 6.375
            var input = State.Create(grammar.Value.InputSymbol, numbers);

            var output = (double[]) ast.Invoke(input);
            Assert.AreEqual(4.975,   output[0], 0.001d);
            Assert.AreEqual(9.075,  output[1], 0.001d);
            Assert.AreEqual(-20.425, output[2], 0.001d);
            Assert.AreEqual(6.375,   output[3], 0.001d);
        }

        [TestMethod]
        public void TestLearnCenter()
        {
            var grammar = DSLCompiler.LoadGrammarFromFile("../../../ProseDSLModels.grammar");
            printGrammar(grammar);
            SynthesisEngine prose = new SynthesisEngine(grammar.Value);

            double[] inp1 = new double[4] {10.9, 15.0, -14.5, 12.3};
            double[] out1 = center(inp1);
            var input = State.Create(grammar.Value.InputSymbol, inp1);
    
            var examples = new Dictionary<State, object> { { input, out1 }};
            var spec = new ExampleSpec(examples);
            var learnedSet = prose.LearnGrammar(spec);
            stdoutprogram(learnedSet.RealizedPrograms.First().ToString(), "Centre");
            var output = (double[]) learnedSet.RealizedPrograms.First().Invoke(input);
            Assert.AreEqual(4.975,   output[0], 0.001d);
            Assert.AreEqual(9.075,  output[1], 0.001d);
            Assert.AreEqual(-20.425, output[2], 0.001d);
            Assert.AreEqual(6.375,   output[3], 0.001d);
            //test 1000 random examples
            for (int i=0;i<1000;++i) {
                var iTmp = genRnd();
                var inpTmp = State.Create(grammar.Value.InputSymbol, iTmp);
                var oTmp = center(iTmp);
                var outTmp = (double[]) learnedSet.RealizedPrograms.First().Invoke(inpTmp);
                TestContext.WriteLine("Excpt [{0}]", string.Join(", ", oTmp));
                TestContext.WriteLine("Actul [{0}]", string.Join(", ", outTmp));
                Assert.AreEqual(oTmp[0], outTmp[0], 0.001d);
                Assert.AreEqual(oTmp[1], outTmp[1], 0.001d);
                Assert.AreEqual(oTmp[2], outTmp[2], 0.001d);
                Assert.AreEqual(oTmp[3], outTmp[3], 0.001d);
            }
        }

        private double[] add(double[] x){
            var ans = x.Aggregate((a, b) => a + b);
            double[] res = new double[4] {Math.Round(ans,4), 0.0, 0.0, 0.0};
            return res;
        }

        private double[] stall(double[] x) {
            return x;
        }

        private double[] mult(double[] x){
            var ans = x.Aggregate((a, b) => a * b);
            double[] res = new double[4] {Math.Round(ans,4), 0.0, 0.0, 0.0};
            return res;
        }

        private double[] avg_val(double[] x){
            var add_v = add(x);
            var len = (double)x.Length;
            var ans = add_v[0] / len;
            double[] res = new double[4] {Math.Round(ans,4), 0.0, 0.0, 0.0};
            return res;
        }

        private double[] center(double[] x){
            var avg = avg_val(x);
            var ans = x.Select(a => Math.Round(a - avg[0],4)).ToArray(); 
            return ans;
        }

        private double[] genRnd(){
          Random random = new Random( );
          double[] arr = new double[4];
          for (int i=0;i<4;++i) {
            int mult =random.Next(-100, 101);
            arr[i] = Math.Round(mult*random.NextDouble(),4);
          }
         return arr;
        }


        private void printGrammar(Result<Grammar> grammar) {
            Console.WriteLine("grammar.HasErrors");
            Console.WriteLine(grammar.HasErrors);

            Console.WriteLine("grammar.Value");
            Console.WriteLine(grammar.Value);

            Console.WriteLine("grammar.Exception");
            Console.WriteLine(grammar.Exception);

            Console.WriteLine("grammar.Diagnostics");
            Console.WriteLine(grammar.Diagnostics);

            Console.WriteLine("grammar.TraceDiagnostics()");
            grammar.TraceDiagnostics();
        }

        private void stdoutprogram(string program,string name){
            using (System.IO.StreamWriter file = 
            new System.IO.StreamWriter(@"../../../programs.log", true))
            {
                file.WriteLine("#"+name);
                file.WriteLine(program);
                file.WriteLine("");
            }
        }

    }
}
