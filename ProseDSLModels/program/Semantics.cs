using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ProgramSynthesis.Utils;

namespace ProseDSLModels {

	public static class Semantics
	{
	    public static double[] add(double[] x) {
	        if (x == null || x.Length == 0)
	            return null;
	        var ans = x.Aggregate((a, b) => a + b);
	        double[] res = new double[4] {Math.Round(ans,4), 0.0, 0.0, 0.0};
	        return res;
	    }

	    public static double[] mult(double[] x) {
	    	if (x == null || x.Length == 0)
	            return null;
	        var ans = x.Aggregate((a, b) => a * b);
	        double[] res = new double[4] {Math.Round(ans,4), 0.0, 0.0, 0.0};
	        return res;
	    }

	    public static double[] len(double[] x) {
	    	if (x == null || x.Length == 0)
	            return null;
	        var ans = (double)x.Length;
	        double[] res = new double[4] {ans, 0.0, 0.0, 0.0};
	        return res;
	    }

	    public static double[] div(double[] x, double[] mem) {
	    	if (x == null || x.Length == 0 || mem == null || mem.Length == 0)
	            return null;
	        var ans = x[0] / mem[0];
	        double[] res = new double[4] {Math.Round(ans,4), 0.0, 0.0, 0.0};
	        return res;
	    }

	    public static double[] sub(double[] x, double[] mem) {
	    	if (x == null || x.Length == 0 || mem == null || mem.Length == 0)
	            return null;
	        var ans = mem.Select(a => Math.Round(a - x[0],4)).ToArray(); 
	        return ans;
	    }

	    public static double[] stall(double[] x) {
	    	return x;
	    }
	}
}