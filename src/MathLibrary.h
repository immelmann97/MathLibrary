#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>

class LeastSquareFitting
{
public:
	// Struct encapsulating the polynomial fit's report
	struct polyFitReport
	{
		std::vector<double> coefficients;
		std::vector<double> evaluatedValues;
		double meanError = 0;
		double standardDeviation = 0;
		double maximumError = 0;
		bool isSuccessful;
	};
	// Polynomial interpolation of order k
	void polynomialFit(const std::vector<double>& indices,
		const std::vector<double>& values,
		polyFitReport& report,
		const size_t order);

	// Evaluate the polynomial's information contained in "report", whether they come from polynomialFit or are user generated, and updates "report"'s field estimatedValues with the result.
	void polinomialEvaluation(const std::vector<double>& indices, polyFitReport& report);
};

