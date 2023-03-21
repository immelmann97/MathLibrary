#include "MathLibrary.h"

//tex:The function implements the least square polynomial fitting.
//It tries to interpolate the vector values defined on the range
//indices with a k-th order polynomial of type
//$$v^* = a_0 + ... + a_k t^k$$
//This is achived by defining the matrix $T$ as
//$$\begin{bmatrix}
//1 & t_1 & t_1^2 & ... & t_1^k \\
//& & ... & & \\
//1 & t_n & t_n^2 & ... & t_n^k
//\end{bmatrix}$$
//where each line is the variable of the polynomial evaluated at index.
//After that, is possible to define the system
//$$V = Ta$$
//where $V$ is the input vector "values".
//Since the system is not invertible as it is, the solution is a least square one given by:
//$$a = \left(T^TT\right)^{-1}T^TV$$
//which is a $k+1$ vector containing the polynomial's coefficients. 
void LeastSquareFitting::polynomialFit(const std::vector<double>& indices, const std::vector<double>& values, polyFitReport& report, const size_t order)
{
	// Check if inidces and values have the same size
	if (indices.size() != values.size())
	{
		report.isSuccessful = false;
	}

	// Check if polynomial's order is smaller or equal than the vector's size
	if (indices.size() < (order + 1))
	{
		report.isSuccessful = false;
	}

	// Allocate utility matrix and vectors
	Eigen::MatrixXd T(indices.size(), order + 1);
	Eigen::VectorXd V = Eigen::VectorXd::Map(&values.front(), values.size());
	Eigen::VectorXd result;

	//tex:Populate coefficient's matrix $T$
	for (size_t i = 0; i < indices.size(); i++)
	{
		for (size_t j = 0; j < order+1; j++)
		{
			T(i, j) = pow(indices.at(i), j);
		}
	}

	//tex:Solve $$a = \left(T^TT\right)^{-1}T^TV$$
	result = T.householderQr().solve(V);

	// Insert the result in the coefficient vector
	report.coefficients.resize(order + 1);
	for (size_t i = 0; i < report.coefficients.size(); i++)
	{
		report.coefficients[i] = result[i];
	}
	report.isSuccessful = true;


}

void LeastSquareFitting::polinomialEvaluation(const std::vector<double>& indices, polyFitReport& report)
{
	Eigen::VectorXd t(indices.data());
	Eigen::VectorXd V;
	V.resize(t.size());
	
	// Evaluate polynomial's value at "indices"
	for (size_t i = 0; i < report.coefficients.size(); i++)
	{
		V += report.coefficients[i] * t.pow(i);
	}

	report.evaluatedValues.resize(indices.size());
	for (size_t i = 0; i < report.evaluatedValues.size(); i++)
	{
		report.evaluatedValues[i] = V[i];
	}
}
